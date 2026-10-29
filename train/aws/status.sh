#!/bin/bash
export AWS_PROFILE=${AWS_PROFILE:-nirty} AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
aws s3 cp s3://${TTC_BUCKET:-talk-to-the-cell-688567285324}/${1:-act-train}/status.txt - 2>/dev/null || echo "no status yet"
