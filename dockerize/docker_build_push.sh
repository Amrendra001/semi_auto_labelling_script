repo=semi_auto_labelling
branch=dev
lambdaname=semi_auto_labelling
USER_ID_DEV=$(aws sts get-caller-identity --profile dev --query "Account" --output text)
image=$USER_ID_DEV.dkr.ecr.ap-south-1.amazonaws.com/$repo:$branch
DEV_REGION=ap-south-1

echo dockerizing $branch to $image
docker build -t $image --platform linux/arm64 -f dockerize/dockerfile .

echo logging into ECR
aws ecr get-login-password \
    --region $DEV_REGION \
    --profile dev \
| docker login \
    --username AWS \
    --password-stdin $USER_ID_DEV.dkr.ecr.$DEV_REGION.amazonaws.com

echo pushing $image
docker push $image

echo deploy on lambda: $lambdaname
aws lambda update-function-code \
    --function-name  $lambdaname \
    --profile dev \
    --image-uri $image
