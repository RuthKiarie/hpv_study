import boto3
import json
from botocore.exceptions import ClientError

# Initialize Boto3 client for Step Functions
sfn_client = boto3.client('stepfunctions', region_name='us-east-1')

# Define your state machine name and IAM Role ARN
STATE_MACHINE_NAME = "HpvPipeline"
ROLE_ARN = "arn:aws:iam::041904915024:role/sm-hpv-processing-role"

def deploy_state_machine():
    # 1. Read the workflow definition from statemachine.json
    with open('statemachine.json', 'r') as f:
        definition_content = f.read()

    # 2. Get all existing state machines to check if yours already exists
    response = sfn_client.list_state_machines()
    existing_arns = {sm['name']: sm['stateMachineArn'] for sm in response.get('stateMachines', [])}

    if STATE_MACHINE_NAME in existing_arns:
        # Update existing state machine
        arn = existing_arns[STATE_MACHINE_NAME]
        print(f"Updating existing state machine: {STATE_MACHINE_NAME} ({arn})...")
        
        sfn_client.update_state_machine(
            stateMachineArn=arn,
            definition=definition_content,
            roleArn=ROLE_ARN
        )
        print("Successfully updated state machine!")
    else:
        # Create a new state machine
        print(f"Creating new state machine: {STATE_MACHINE_NAME}...")
        
        response = sfn_client.create_state_machine(
            name=STATE_MACHINE_NAME,
            definition=definition_content,
            roleArn=ROLE_ARN,
            type='STANDARD'
        )
        print(f"Successfully created state machine! ARN: {response['stateMachineArn']}")

if __name__ == "__main__":
    deploy_state_machine()