#!/bin/bash

# Sports Video MLOps - AWS Free Tier Setup Script
# This script sets up the entire AWS infrastructure for free tier deployment

set -e

echo "🚀 Setting up Sports Video MLOps on AWS Free Tier..."

# Configuration
APP_NAME="sports-video-api"
AWS_REGION="${AWS_REGION:-us-east-1}"
KEY_NAME="${KEY_NAME:-sports-mlops-key}"
INSTANCE_TYPE="t3.micro"  # Free tier eligible

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check AWS CLI
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found. Please install it first."
        exit 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Run 'aws configure' first."
        exit 1
    fi
    
    log_info "AWS CLI configured ✓"
}

# Create ECR repository
create_ecr_repo() {
    log_info "Creating ECR repository..."
    
    if aws ecr describe-repositories --repository-names $APP_NAME &> /dev/null; then
        log_warn "ECR repository already exists"
    else
        aws ecr create-repository \
            --repository-name $APP_NAME \
            --region $AWS_REGION \
            --image-scanning-configuration scanOnPush=true
        log_info "ECR repository created ✓"
    fi
    
    # Get ECR URI
    ECR_URI=$(aws ecr describe-repositories \
        --repository-names $APP_NAME \
        --region $AWS_REGION \
        --query 'repositories[0].repositoryUri' \
        --output text)
    
    echo "ECR_URI=$ECR_URI" >> aws_config.env
}

# Create security group
create_security_group() {
    log_info "Creating security group..."
    
    # Check if security group exists
    if aws ec2 describe-security-groups --group-names $APP_NAME &> /dev/null; then
        SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
            --group-names $APP_NAME \
            --query 'SecurityGroups[0].GroupId' \
            --output text)
        log_warn "Security group already exists: $SECURITY_GROUP_ID"
    else
        SECURITY_GROUP_ID=$(aws ec2 create-security-group \
            --group-name $APP_NAME \
            --description "Security group for Sports Video API" \
            --query 'GroupId' \
            --output text)
        
        # Add inbound rules
        aws ec2 authorize-security-group-ingress \
            --group-id $SECURITY_GROUP_ID \
            --protocol tcp \
            --port 80 \
            --cidr 0.0.0.0/0
        
        aws ec2 authorize-security-group-ingress \
            --group-id $SECURITY_GROUP_ID \
            --protocol tcp \
            --port 22 \
            --cidr 0.0.0.0/0
        
        log_info "Security group created: $SECURITY_GROUP_ID ✓"
    fi
    
    echo "SECURITY_GROUP_ID=$SECURITY_GROUP_ID" >> aws_config.env
}

# Create key pair
create_key_pair() {
    log_info "Creating EC2 key pair..."
    
    if aws ec2 describe-key-pairs --key-names $KEY_NAME &> /dev/null; then
        log_warn "Key pair already exists"
    else
        aws ec2 create-key-pair \
            --key-name $KEY_NAME \
            --query 'KeyMaterial' \
            --output text > $KEY_NAME.pem
        
        chmod 400 $KEY_NAME.pem
        log_info "Key pair created: $KEY_NAME.pem ✓"
    fi
}

# Launch EC2 instance
launch_ec2_instance() {
    log_info "Launching EC2 instance..."
    
    # Get latest Amazon Linux 2 AMI
    AMI_ID=$(aws ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
        --query 'Images|sort_by(@, &CreationDate)[-1].ImageId' \
        --output text)
    
    # User data script for instance setup
    cat > user-data.sh << 'EOF'
#!/bin/bash
yum update -y
yum install -y docker

# Start Docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
rpm -U ./amazon-cloudwatch-agent.rpm

# Create application directory
mkdir -p /app
chown ec2-user:ec2-user /app
EOF

    # Launch instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id $AMI_ID \
        --count 1 \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-groups $APP_NAME \
        --user-data file://user-data.sh \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$APP_NAME}]" \
        --query 'Instances[0].InstanceId' \
        --output text)
    
    log_info "EC2 instance launched: $INSTANCE_ID ✓"
    
    # Wait for instance to be running
    log_info "Waiting for instance to be running..."
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID
    
    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    echo "INSTANCE_ID=$INSTANCE_ID" >> aws_config.env
    echo "PUBLIC_IP=$PUBLIC_IP" >> aws_config.env
    
    log_info "Instance is running at: http://$PUBLIC_IP"
}

# Create CloudWatch log group
create_cloudwatch_logs() {
    log_info "Creating CloudWatch log group..."
    
    if aws logs describe-log-groups --log-group-name-prefix "/aws/ec2/$APP_NAME" | grep -q "$APP_NAME"; then
        log_warn "CloudWatch log group already exists"
    else
        aws logs create-log-group \
            --log-group-name "/aws/ec2/$APP_NAME" \
            --region $AWS_REGION
        log_info "CloudWatch log group created ✓"
    fi
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    # Login to ECR
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI
    
    # Build image
    docker build -t $APP_NAME .
    docker tag $APP_NAME:latest $ECR_URI:latest
    
    # Push image
    docker push $ECR_URI:latest
    
    log_info "Docker image pushed to ECR ✓"
}

# Deploy application
deploy_application() {
    log_info "Deploying application to EC2..."
    
    # Create deployment script
    cat > deploy-remote.sh << EOF
#!/bin/bash
# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI

# Stop existing container
docker stop $APP_NAME 2>/dev/null || true
docker rm $APP_NAME 2>/dev/null || true

# Pull and run new container
docker pull $ECR_URI:latest
docker run -d \\
    --name $APP_NAME \\
    -p 80:8000 \\
    -e AWS_REGION=$AWS_REGION \\
    -e ENVIRONMENT=production \\
    --restart unless-stopped \\
    $ECR_URI:latest

echo "Application deployed successfully!"
EOF

    # Copy and execute script on EC2
    scp -i $KEY_NAME.pem -o StrictHostKeyChecking=no deploy-remote.sh ec2-user@$PUBLIC_IP:/tmp/
    ssh -i $KEY_NAME.pem -o StrictHostKeyChecking=no ec2-user@$PUBLIC_IP "chmod +x /tmp/deploy-remote.sh && /tmp/deploy-remote.sh"
    
    log_info "Application deployed successfully! ✓"
    log_info "Access your API at: http://$PUBLIC_IP"
}

# Main execution
main() {
    log_info "Starting AWS setup for Sports Video MLOps..."
    
    # Initialize config file
    echo "# AWS Configuration" > aws_config.env
    echo "AWS_REGION=$AWS_REGION" >> aws_config.env
    
    check_aws_cli
    create_ecr_repo
    create_security_group
    create_key_pair
    launch_ec2_instance
    create_cloudwatch_logs
    
    # Wait a bit for instance to be fully ready
    log_info "Waiting for instance to be fully ready..."
    sleep 60
    
    build_and_push_image
    deploy_application
    
    log_info "🎉 Setup complete!"
    log_info "Your Sports Video API is running at: http://$PUBLIC_IP"
    log_info "Test it with: curl http://$PUBLIC_IP/health"
    
    # Show configuration
    echo ""
    echo "=== Configuration ==="
    cat aws_config.env
    echo ""
    
    # Cleanup
    rm -f user-data.sh deploy-remote.sh
}

# Run main function
main "$@"
