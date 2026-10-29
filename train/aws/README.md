# ACT training and the demo host on AWS (Intel Xeon, c7i)

Scripts used to train every ACT round in `docs/results.md` and to host the demo. They use your AWS
profile (`AWS_PROFILE`, default `nirty`) and bucket (`TTC_BUCKET`).

| Script | What it does |
|---|---|
| `ship_dataset.sh PREFIX` | Convert `runs/dataset` (or `RAW=...`) to LeRobot and upload it to `s3://$TTC_BUCKET/PREFIX/` |
| `make_userdata.py PREFIX MAX_TIME WORKERS` | Boot script for a training box: installs physicalai-train, trains ACT (bf16), uploads the checkpoint and a status line |
| `launch.sh PREFIX [TYPE]` | Start the instance (default c7i.4xlarge); it terminates itself when training ends |
| `status.sh PREFIX` | Read the instance's status line from S3 |
| `eval_policy.sh CKPT NAME TRIALS SEED DEMOS` | Export a checkpoint to OpenVINO and score it in the local cell |
| `make_demo_userdata.py` | Boot script for a demo host: Docker build + the whole cell (then `launch.sh demo c7i.2xlarge`) |

Generated `user-data-*.sh` files contain time-limited signed S3 links and are git-ignored.
