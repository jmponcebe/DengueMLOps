#!/usr/bin/env bash
# ============================================================
# aws-teardown.sh — Eliminar toda la infraestructura AWS
# Uso: ./scripts/aws-teardown.sh
# ============================================================
set -euo pipefail

REGION="us-east-1"
STACK_NAME="dengue-mlops"

echo "=== Dengue MLOps — Teardown AWS ==="
echo "[WARN] Esto eliminará TODOS los recursos del stack $STACK_NAME"
read -p "¿Continuar? (y/N): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
  echo "Cancelado."
  exit 0
fi

# --- Limpiar imágenes ECR antes de borrar repos ---
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || true)

# --- Vaciar bucket S3 (requerido antes de borrar stack) ---
S3_BUCKET=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" \
  --output text --region "$REGION" 2>/dev/null || true)
if [ -n "$S3_BUCKET" ] && [ "$S3_BUCKET" != "None" ]; then
  echo "[INFO] Vaciando bucket S3: $S3_BUCKET"
  aws s3 rm "s3://$S3_BUCKET" --recursive --region "$REGION" 2>/dev/null || true
fi

for REPO in dengue-api dengue-dashboard; do
  echo "[INFO] Limpiando imágenes de $REPO..."
  IMAGES=$(aws ecr list-images --repository-name "$REPO" \
    --query "imageIds[*]" --output json --region "$REGION" 2>/dev/null || echo "[]")
  if [ "$IMAGES" != "[]" ] && [ "$IMAGES" != "" ]; then
    aws ecr batch-delete-image --repository-name "$REPO" \
      --image-ids "$IMAGES" --region "$REGION" 2>/dev/null || true
  fi
done

# --- Escalar servicios a 0 antes de borrar ---
echo "[INFO] Escalando servicios a 0..."
for SERVICE in dengue-api-service dengue-dashboard-service; do
  aws ecs update-service --cluster dengue-cluster --service "$SERVICE" \
    --desired-count 0 --region "$REGION" 2>/dev/null || true
done
sleep 5

# --- Borrar stack CloudFormation ---
echo "[INFO] Eliminando stack CloudFormation: $STACK_NAME"
aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$REGION"

echo "[INFO] Esperando a que el stack se elimine..."
aws cloudformation wait stack-delete-complete \
  --stack-name "$STACK_NAME" --region "$REGION" 2>/dev/null || true

# --- Limpiar log groups (por si quedan huérfanos) ---
echo "[INFO] Limpiando log groups..."
for LG in /ecs/dengue-api /ecs/dengue-dashboard; do
  aws logs delete-log-group --log-group-name "$LG" --region "$REGION" 2>/dev/null || true
done

echo ""
echo "=== Teardown completado ==="
echo "Recursos eliminados. Verifica en la consola AWS que no quede nada activo."
