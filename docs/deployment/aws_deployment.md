# Guía de Despliegue en AWS — Learner Lab

## Prerrequisitos

- AWS Learner Lab activo (us-east-1)
- Docker Desktop instalado localmente
- AWS CLI configurado con credenciales del lab
- Imágenes Docker construidas localmente

## 1. Configurar AWS CLI

Desde la consola de AWS Academy, copia las credenciales temporales:

```bash
aws configure
# AWS Access Key ID: <del lab>
# AWS Secret Access Key: <del lab>
# Default region: us-east-1
# Default output: json

# O exportar directamente (credenciales temporales del lab)
export AWS_ACCESS_KEY_ID=<...>
export AWS_SECRET_ACCESS_KEY=<...>
export AWS_SESSION_TOKEN=<...>
```

## 2. Crear repositorios en ECR

```bash
# Crear repositorios para las imágenes
aws ecr create-repository --repository-name dengue-api --region us-east-1
aws ecr create-repository --repository-name dengue-dashboard --region us-east-1
```

## 3. Build y Push de Imágenes

```bash
# Login en ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build
docker build -f Dockerfile.api -t dengue-api:latest .
docker build -f Dockerfile.dashboard -t dengue-dashboard:latest .

# Tag
docker tag dengue-api:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/dengue-api:latest
docker tag dengue-dashboard:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/dengue-dashboard:latest

# Push
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/dengue-api:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/dengue-dashboard:latest
```

## 4. Opción A: Despliegue en EC2 (Más simple)

Esta opción es la más sencilla para el TFM y consume menos presupuesto.

### Lanzar instancia EC2

```bash
# Lanzar EC2 t2.medium con Amazon Linux 2
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t2.medium \
  --key-name vockey \
  --security-group-ids <sg-id> \
  --iam-instance-profile Name=LabInstanceProfile \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=dengue-mlops}]' \
  --user-data file://scripts/ec2-userdata.sh
```

### Security Group (abrir puertos)

```bash
aws ec2 authorize-security-group-ingress \
  --group-id <sg-id> \
  --protocol tcp --port 8000 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id <sg-id> \
  --protocol tcp --port 8501 --cidr 0.0.0.0/0
```

### Conectar y desplegar

```bash
ssh -i labsuser.pem ec2-user@<PUBLIC-IP>

# Instalar Docker
sudo yum update -y
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Login en ECR y pull
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
docker pull <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/dengue-api:latest
docker pull <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/dengue-dashboard:latest

# Ejecutar
docker run -d -p 8000:8000 --name api <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/dengue-api:latest
docker run -d -p 8501:8501 -e API_URL=http://<PRIVATE-IP>:8000 --name dashboard <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/dengue-dashboard:latest
```

Acceder: `http://<PUBLIC-IP>:8501` (dashboard) y `http://<PUBLIC-IP>:8000/docs` (API Swagger)

## 4. Opción B: Despliegue en ECS/Fargate (Más profesional)

Mejor para demostrar competencias MLOps en el TFM.

### Crear cluster ECS

```bash
aws ecs create-cluster --cluster-name dengue-cluster
```

### Task Definition (API)

```json
{
  "family": "dengue-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/LabRole",
  "taskRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/LabRole",
  "containerDefinitions": [
    {
      "name": "dengue-api",
      "image": "<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/dengue-api:latest",
      "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/dengue-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

```bash
aws ecs register-task-definition --cli-input-json file://aws/task-def-api.json
```

### Crear servicio ECS

```bash
aws ecs create-service \
  --cluster dengue-cluster \
  --service-name dengue-api-service \
  --task-definition dengue-api \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-id>],securityGroups=[<sg-id>],assignPublicIp=ENABLED}"
```

## 5. Verificación

```bash
# Health check
curl http://<HOST>:8000/health

# Predicción de prueba
curl -X POST http://<HOST>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "month_sin": 0.5, "month_cos": 0.866,
    "se_sin": 0.82, "se_cos": -0.57,
    "is_peak_season": 1, "quarter": 1,
    "tempmed_lag8w": 26.0, "tempmed_roll12w": 25.5,
    "umidmed_roll4w": 80.0, "temp_x_humid_lag4w": 2080.0,
    "pop_log": 12.0,
    "region_Nordeste": 0, "region_Centro-Oeste": 0,
    "region_Sudeste": 1, "region_Sul": 0
  }'
```

## Costes estimados (Learner Lab)

| Recurso | Coste/hora aprox. |
| --- | --- |
| EC2 t2.medium | ~$0.046 |
| Fargate (0.5 vCPU, 1GB) | ~$0.025 |
| ECR (almacenamiento) | ~$0.10/GB/mes |
| S3 (artifacts) | Despreciable |

**Recomendación**: Usar EC2 para demos del TFM (más control, más barato).
Detener la instancia cuando no se use para conservar presupuesto.

## Limpieza

```bash
# Parar instancias EC2
aws ec2 stop-instances --instance-ids <instance-id>

# O eliminar servicios ECS
aws ecs update-service --cluster dengue-cluster --service dengue-api-service --desired-count 0
aws ecs delete-service --cluster dengue-cluster --service dengue-api-service
aws ecs delete-cluster --cluster dengue-cluster

# Limpiar imágenes ECR
aws ecr batch-delete-image --repository-name dengue-api --image-ids imageTag=latest
aws ecr batch-delete-image --repository-name dengue-dashboard --image-ids imageTag=latest
```
