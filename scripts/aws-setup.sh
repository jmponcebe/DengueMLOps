#!/usr/bin/env bash
# ============================================================
# aws-setup.sh — Despliegue completo de infraestructura + imágenes
# Uso: ./scripts/aws-setup.sh [--skip-images]
# ============================================================
set -euo pipefail

REGION="us-east-1"
STACK_NAME="dengue-mlops"
TEMPLATE="aws/cloudformation.yml"
SKIP_IMAGES=false

for arg in "$@"; do
  case $arg in
    --skip-images) SKIP_IMAGES=true ;;
  esac
done

echo "=== Dengue MLOps — Setup AWS ==="

# --- Verificar credenciales ---
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "[INFO] Cuenta AWS: $ACCOUNT_ID"
echo "[INFO] Región: $REGION"

# --- Obtener VPC y Subnet por defecto ---
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text --region "$REGION")
SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" \
  --query "Subnets[0].SubnetId" --output text --region "$REGION")
echo "[INFO] VPC: $VPC_ID | Subnet: $SUBNET_ID"

# --- Obtener LabRole ARN ---
ROLE_ARN=$(aws iam get-role --role-name LabRole --query "Role.Arn" --output text 2>/dev/null || true)
if [ -z "$ROLE_ARN" ]; then
  echo "[WARN] LabRole no encontrado, usando ecsTaskExecutionRole"
  ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/ecsTaskExecutionRole"
fi
echo "[INFO] Execution Role: $ROLE_ARN"

# ============================================================
# Fase 1: Desplegar stack base (solo ECR + SG + Cluster)
# ============================================================
echo ""
echo "=== Fase 1: Desplegando infraestructura base ==="

aws cloudformation deploy \
  --template-file "$TEMPLATE" \
  --stack-name "$STACK_NAME" \
  --parameter-overrides \
    VpcId="$VPC_ID" \
    SubnetId="$SUBNET_ID" \
    ExecutionRoleArn="$ROLE_ARN" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "$REGION" \
  --no-fail-on-empty-changeset

echo "[OK] Stack base desplegado"

# --- Obtener URIs de ECR y nombre del bucket S3 ---
API_REPO=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='ECRRepoApiUri'].OutputValue" \
  --output text --region "$REGION")
DASH_REPO=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='ECRRepoDashboardUri'].OutputValue" \
  --output text --region "$REGION")
S3_BUCKET=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" \
  --output text --region "$REGION")
echo "[INFO] ECR API: $API_REPO"
echo "[INFO] ECR Dashboard: $DASH_REPO"
echo "[INFO] S3 Bucket: $S3_BUCKET"

# ============================================================
# Fase 2: Subir datos y modelo a S3
# ============================================================
echo ""
echo "=== Fase 2: Subiendo datos y modelo a S3 ==="

# Modelo champion
echo "[INFO] Subiendo modelo champion..."
aws s3 sync mlflow-artifacts/models/champion/ "s3://$S3_BUCKET/models/champion/" \
  --region "$REGION" --quiet

# Datos parquet
echo "[INFO] Subiendo datos parquet..."
for DIR in data/raw data/interim data/processed; do
  if [ -d "$DIR" ]; then
    aws s3 sync "$DIR/" "s3://$S3_BUCKET/$DIR/" --exclude "*.gitkeep" \
      --region "$REGION" --quiet
  fi
done

# GeoJSON
if [ -f "data/external/brazil_uf.geojson" ]; then
  echo "[INFO] Subiendo GeoJSON..."
  aws s3 cp data/external/brazil_uf.geojson "s3://$S3_BUCKET/data/external/brazil_uf.geojson" \
    --region "$REGION" --quiet
fi

echo "[OK] Datos y modelo subidos a S3"

# ============================================================
# Fase 3: Build & Push imágenes a ECR
# ============================================================
if [ "$SKIP_IMAGES" = false ]; then
  echo ""
  echo "=== Fase 3: Push de imágenes a ECR ==="

  aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

  # API
  echo "[INFO] Tagueando y subiendo imagen API..."
  docker tag dengue-api:latest "$API_REPO:latest"
  docker push "$API_REPO:latest"

  # Dashboard
  echo "[INFO] Tagueando y subiendo imagen Dashboard..."
  docker tag dengue-dashboard:latest "$DASH_REPO:latest"
  docker push "$DASH_REPO:latest"

  echo "[OK] Imágenes subidas a ECR"

  # ============================================================
  # Fase 4: Actualizar stack con imágenes (crea task defs + services)
  # ============================================================
  echo ""
  echo "=== Fase 4: Desplegando servicios ECS ==="

  aws cloudformation deploy \
    --template-file "$TEMPLATE" \
    --stack-name "$STACK_NAME" \
    --parameter-overrides \
      VpcId="$VPC_ID" \
      SubnetId="$SUBNET_ID" \
      ExecutionRoleArn="$ROLE_ARN" \
      ApiImageUri="$API_REPO:latest" \
      DashboardImageUri="$DASH_REPO:latest" \
    --capabilities CAPABILITY_NAMED_IAM \
    --region "$REGION" \
    --no-fail-on-empty-changeset

  echo "[OK] Servicios ECS desplegados"

  # --- Esperar a que los servicios estén estables ---
  echo "[INFO] Esperando a que los servicios arranquen..."
  aws ecs wait services-stable \
    --cluster dengue-cluster \
    --services dengue-api-service dengue-dashboard-service \
    --region "$REGION" 2>/dev/null || echo "[WARN] Timeout esperando servicios (puede tardar unos minutos)"

  # --- Obtener IPs públicas ---
  echo ""
  echo "=== IPs de los servicios ==="
  for SERVICE in dengue-api-service dengue-dashboard-service; do
    TASK_ARN=$(aws ecs list-tasks --cluster dengue-cluster --service-name "$SERVICE" \
      --query "taskArns[0]" --output text --region "$REGION" 2>/dev/null)
    if [ "$TASK_ARN" != "None" ] && [ -n "$TASK_ARN" ]; then
      ENI=$(aws ecs describe-tasks --cluster dengue-cluster --tasks "$TASK_ARN" \
        --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" \
        --output text --region "$REGION")
      IP=$(aws ec2 describe-network-interfaces --network-interface-ids "$ENI" \
        --query "NetworkInterfaces[0].Association.PublicIp" --output text --region "$REGION" 2>/dev/null)
      echo "  $SERVICE → $IP"
    fi
  done
fi

echo ""
echo "=== Setup completado ==="
echo "Recuerda ejecutar ./scripts/aws-teardown.sh al terminar para evitar costes."
