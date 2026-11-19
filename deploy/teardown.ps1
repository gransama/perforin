# Borra todo lo creado para el despliegue en AWS (ECS Express Mode). Region: us-west-2.
# Uso:  .\deploy\teardown.ps1
$ErrorActionPreference = 'Continue'
aws ecs delete-express-gateway-service --service-arn arn:aws:ecs:us-west-2:543584168314:service/default/wellprofit
Write-Host "Esperando a que ECS elimine el servicio y su balanceador (puede tardar unos minutos)..."
Start-Sleep 120
aws iam detach-role-policy --role-name wellprofit-ecs-exec --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam delete-role --role-name wellprofit-ecs-exec
aws iam detach-role-policy --role-name wellprofit-ecs-infra --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices
aws iam delete-role --role-name wellprofit-ecs-infra
aws logs delete-log-group --log-group-name /aws/ecs/default/wellprofit-fc71
