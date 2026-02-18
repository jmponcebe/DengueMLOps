#!/usr/bin/env bash
# ============================================================
# ec2-deploy.sh — Desplegar en EC2 con Docker Compose
# Alternativa más económica a ECS Fargate para Learner Lab
# Uso: ./scripts/ec2-deploy.sh <IP_EC2>
# ============================================================
set -euo pipefail

REGION="us-east-1"
KEY_NAME="vockey"
KEY_FILE="$HOME/labsuser.pem"

if [ -z "${1:-}" ]; then
  echo "Uso: $0 <IP_EC2>"
  echo ""
  echo "Pasos previos:"
  echo "  1. Lanzar instancia EC2 (t2.medium, Amazon Linux 2023, SG del stack)"
  echo "  2. Descargar labsuser.pem del Learner Lab"
  echo "  3. Ejecutar este script con la IP pública"
  exit 1
fi

EC2_IP="$1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Obtener nombre del bucket S3
S3_BUCKET=$(aws cloudformation describe-stacks --stack-name dengue-mlops \
  --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" \
  --output text --region "$REGION" 2>/dev/null || echo "dengue-mlops-data-$ACCOUNT_ID")

echo "=== Dengue MLOps — Deploy en EC2 ==="
echo "[INFO] EC2: $EC2_IP"
echo "[INFO] Cuenta: $ACCOUNT_ID"
echo "[INFO] Bucket S3: $S3_BUCKET"

# --- Configurar SSH ---
chmod 400 "$KEY_FILE" 2>/dev/null || true
SSH="ssh -i $KEY_FILE -o StrictHostKeyChecking=no ec2-user@$EC2_IP"
SCP="scp -i $KEY_FILE -o StrictHostKeyChecking=no"

# --- Instalar Docker en EC2 ---
echo "[INFO] Instalando Docker en EC2..."
$SSH << 'REMOTE'
sudo dnf update -y -q
sudo dnf install -y docker
sudo systemctl enable docker && sudo systemctl start docker
sudo usermod -aG docker ec2-user

# Docker Compose plugin
DOCKER_COMPOSE_VERSION="v2.27.0"
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL "https://github.com/docker/compose/releases/download/$DOCKER_COMPOSE_VERSION/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
REMOTE

# --- Login en ECR ---
echo "[INFO] Login en ECR desde EC2..."
ECR_PASS=$(aws ecr get-login-password --region "$REGION")
$SSH "docker login --username AWS --password '$ECR_PASS' $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# --- Pull de imágenes ---
API_IMG="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/dengue-api:latest"
DASH_IMG="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/dengue-dashboard:latest"

echo "[INFO] Descargando imágenes..."
$SSH "docker pull $API_IMG && docker pull $DASH_IMG"

# --- Crear docker-compose para EC2 ---
echo "[INFO] Configurando servicios..."
$SSH << REMOTE
cat > ~/docker-compose.yml << 'EOF'
services:
  api:
    image: $API_IMG
    ports:
      - "8000:8000"
    environment:
      - S3_DATA_BUCKET=$S3_BUCKET
      - AWS_DEFAULT_REGION=$REGION
    command: ["/bin/bash", "/app/entrypoint.sh"]
    restart: unless-stopped

  dashboard:
    image: $DASH_IMG
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
      - S3_DATA_BUCKET=$S3_BUCKET
      - AWS_DEFAULT_REGION=$REGION
    command: ["/bin/bash", "/app/entrypoint.sh"]
    depends_on:
      - api
    restart: unless-stopped
EOF

docker compose up -d
REMOTE

echo ""
echo "=== Deploy EC2 completado ==="
echo "  API:       http://$EC2_IP:8000"
echo "  Dashboard: http://$EC2_IP:8501"
echo "  Health:    http://$EC2_IP:8000/health"
