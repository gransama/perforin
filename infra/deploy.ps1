# Despliega WellProfitAI en AWS: S3 + Lambda + Bedrock Agent.
# Requisitos: AWS CLI v2 con credenciales configuradas y acceso al modelo de Bedrock habilitado.
# Uso (desde la raiz del repo):
#   .\infra\deploy.ps1 -Region us-east-1
param(
    [string]$Region = "us-east-1",
    [string]$StackName = "wellprofitai",
    [string]$FoundationModel = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$account = aws sts get-caller-identity --query Account --output text
if ($LASTEXITCODE -ne 0) { throw "AWS CLI sin credenciales validas" }
$artifacts = "$StackName-artifacts-$account-$Region"
Write-Host "Cuenta $account, region $Region, stack $StackName"

# 1. Datos JSON
python scripts/build_api_data.py
if ($LASTEXITCODE -ne 0) { throw "Fallo build_api_data.py" }

# 2. Bucket de artefactos para el codigo de Lambda (se crea una sola vez)
aws s3api head-bucket --bucket $artifacts --region $Region 2>$null
if ($LASTEXITCODE -ne 0) {
    if ($Region -eq "us-east-1") { aws s3api create-bucket --bucket $artifacts --region $Region | Out-Null }
    else { aws s3api create-bucket --bucket $artifacts --region $Region --create-bucket-configuration LocationConstraint=$Region | Out-Null }
    aws s3api put-public-access-block --bucket $artifacts --region $Region `
        --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
}

# 3. Empaquetar y desplegar el stack
New-Item -ItemType Directory -Force build | Out-Null
aws cloudformation package --template-file infra/template.yaml --s3-bucket $artifacts `
    --output-template-file build/packaged.yaml --region $Region
if ($LASTEXITCODE -ne 0) { throw "Fallo cloudformation package" }
aws cloudformation deploy --template-file build/packaged.yaml --stack-name $StackName `
    --capabilities CAPABILITY_IAM --region $Region `
    --parameter-overrides FoundationModel=$FoundationModel
if ($LASTEXITCODE -ne 0) { throw "Fallo cloudformation deploy" }

# 4. Subir los datos al bucket del stack
$bucket = aws cloudformation describe-stacks --stack-name $StackName --region $Region `
    --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" --output text
aws s3 sync build/data "s3://$bucket/" --delete --region $Region
if ($LASTEXITCODE -ne 0) { throw "Fallo la subida a S3" }

aws cloudformation describe-stacks --stack-name $StackName --region $Region `
    --query "Stacks[0].Outputs" --output table
