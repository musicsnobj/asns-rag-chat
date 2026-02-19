# 1. Set your AWS region and account ID

export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPO_NAME=asns-rag-chat-image
export REPO_URI=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:latest

# 2. Create an ECR repository (if it doesn't exist)

aws ecr create-repository \
 --repository-name ${REPO_NAME} \
 --region ${AWS_REGION}

# 3. Authenticate Docker to ECR

aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# 4. Build your Docker image

docker build -t ${REPO_NAME} .

# 5. Tag the image

docker tag ${REPO_NAME}:latest ${REPO_URI}

# 6. Push to ECR

docker push ${REPO_URI}

# 7. Update your Lambda function to use the container image

aws lambda update-function-code \
 --function-name asns-rag-chat \
 --image-uri ${REPO_URI}
