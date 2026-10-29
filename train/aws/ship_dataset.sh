#!/bin/bash
# Convert the successful recorded episodes to LeRobot and hand them to the training box via S3.
set -e
S=$(cd "$(dirname "$0")" && pwd)
DATASET=${DATASET:-panda_part_flip}; RAW=${RAW:-runs/dataset}   # e.g. DATASET=panda_soup_can RAW=runs/dataset_can
cd "$(dirname "$0")/../.."
nice -n 10 ${TRAIN_PYTHON:-python} -m v3.convert_lerobot $RAW --out datasets/$DATASET --repo-id local/$DATASET 2>&1 | grep -E "^wrote|Error|error"
du -sh datasets/$DATASET
nice -n 10 tar czf $S/dataset.tgz -C datasets $DATASET
ls -la $S/dataset.tgz
export AWS_PROFILE=${AWS_PROFILE:-nirty} AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
aws s3 cp --only-show-errors $S/dataset.tgz s3://${TTC_BUCKET:-talk-to-the-cell-688567285324}/${1:-act-train}/dataset.tgz && echo uploaded
