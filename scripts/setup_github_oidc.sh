#!/usr/bin/env bash
# One-time setup: lets GitHub Actions deploy to this AWS account WITHOUT long-lived access keys,
# using OIDC federation. The role can only be assumed by workflows from YOUR repo, on the main
# branch or the "production" environment, and can only assume the CDK bootstrap roles
# (plus invoke the API Lambda and run the state machine for the post-deploy checks).
#
# Usage:  GH_REPO=<owner>/<repo> ./scripts/setup_github_oidc.sh
# Prereq: `cdk bootstrap` has been run once in this account/region.
set -euo pipefail

: "${GH_REPO:?Set GH_REPO, e.g. GH_REPO=RuthKiarie/hpv-mlops}"
ROLE_NAME="${ROLE_NAME:-github-actions-hpv-mlops-deploy}"
REGION="${AWS_REGION:-us-east-1}"
QUALIFIER="${CDK_QUALIFIER:-hnb659fds}"   # CDK's default bootstrap qualifier
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
OIDC_HOST="token.actions.githubusercontent.com"
OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/${OIDC_HOST}"

if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" > /dev/null 2>&1; then
  echo "OIDC provider already exists."
else
  aws iam create-open-id-connect-provider --url "https://${OIDC_HOST}" \
    --client-id-list sts.amazonaws.com \
    --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 > /dev/null
  echo "Created OIDC provider."
fi

# Two "sub" claims: jobs without an environment (plan) carry the branch ref; the deploy job
# carries the environment name instead. Both must be allowed.
TRUST=$(cat <<JSON
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Federated": "${OIDC_ARN}"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {"${OIDC_HOST}:aud": "sts.amazonaws.com"},
      "StringLike": {"${OIDC_HOST}:sub": [
        "repo:${GH_REPO}:ref:refs/heads/main",
        "repo:${GH_REPO}:environment:production"
      ]}
    }
  }]
}
JSON
)

PERMISSIONS=$(cat <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "AssumeCdkBootstrapRoles", "Effect": "Allow", "Action": "sts:AssumeRole",
     "Resource": "arn:aws:iam::${ACCOUNT_ID}:role/cdk-${QUALIFIER}-*"},
    {"Sid": "ReadBootstrapVersion", "Effect": "Allow", "Action": "ssm:GetParameter",
     "Resource": "arn:aws:ssm:${REGION}:${ACCOUNT_ID}:parameter/cdk-bootstrap/${QUALIFIER}/version"},
    {"Sid": "SmokeTestApi", "Effect": "Allow", "Action": "lambda:InvokeFunction",
     "Resource": "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:HpvMlopsStackV2-*"},
    {"Sid": "RunPipeline", "Effect": "Allow", "Action": ["states:StartExecution", "states:DescribeExecution"],
     "Resource": [
       "arn:aws:states:${REGION}:${ACCOUNT_ID}:stateMachine:hpv-mlops-pipeline-v2",
       "arn:aws:states:${REGION}:${ACCOUNT_ID}:execution:hpv-mlops-pipeline-v2:*"
     ]}
  ]
}
JSON
)

if 
aws iam get-role --role-name "$ROLE_NAME" > /dev/null 2>&1; then
  aws iam update-assume-role-policy --role-name "$ROLE_NAME" --policy-document "$TRUST"
  echo "Updated trust policy on existing role."
else
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "$TRUST" > /dev/null
  echo "Created role."
fi

aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name deploy-permissions --policy-document "$PERMISSIONS"

ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)
echo
echo "Done. Add this as a GitHub repository VARIABLE (Settings > Secrets and variables > Actions > Variables):"
echo "  AWS_DEPLOY_ROLE_ARN = ${ROLE_ARN}"
