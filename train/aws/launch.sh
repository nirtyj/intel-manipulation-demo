#!/bin/bash
# launch.sh PREFIX [INSTANCE_TYPE]  -> an Intel Xeon training box running user-data-PREFIX.sh
S=$(cd "$(dirname "$0")" && pwd); export AWS_PROFILE=${AWS_PROFILE:-nirty} AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
aws ec2 run-instances \
  --image-id resolve:ssm:/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  --instance-type ${2:-c7i.4xlarge} --key-name talk-to-the-cell --security-group-ids sg-0fce63c4d9dc75671 \
  --instance-initiated-shutdown-behavior terminate --metadata-options HttpTokens=required \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=60,VolumeType=gp3,DeleteOnTermination=true}' \
  --user-data file://$S/user-data-$1.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=talk-to-the-cell-$1},{Key=project,Value=talk-to-the-cell}]" 'ResourceType=volume,Tags=[{Key=project,Value=talk-to-the-cell}]' \
  --query 'Instances[0].InstanceId' --output text
