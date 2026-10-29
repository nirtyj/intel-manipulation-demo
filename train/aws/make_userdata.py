import json, os, sys, boto3
S = os.path.dirname(os.path.abspath(__file__))
bucket = os.environ.get("TTC_BUCKET", "talk-to-the-cell-688567285324")
prefix = sys.argv[1] if len(sys.argv) > 1 else "act-train"
max_time = sys.argv[2] if len(sys.argv) > 2 else "00:01:30:00"
workers = sys.argv[3] if len(sys.argv) > 3 else "10"
dataset = os.environ.get("DATASET", "panda_part_flip")  # panda_soup_can for the YCB can
s3 = boto3.Session(profile_name="nirty", region_name="us-east-1").client("s3")
ttl = 12 * 3600
urls = {"DATA_URL": s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": f"{prefix}/dataset.tgz"}, ExpiresIn=ttl)}
for name, key in (("CKPT_URL", "last.ckpt"), ("LOGS_URL", "logs.tgz"), ("STATUS_URL", "status.txt")):
    urls[name] = s3.generate_presigned_url("put_object", Params={"Bucket": bucket, "Key": f"{prefix}/{key}"}, ExpiresIn=ttl)
json.dump({"bucket": bucket, "prefix": prefix}, open(f"{S}/s3.json", "w"))
cfg = open(os.path.join(S, "../../v3/act_cpu.yaml")).read()
cfg = cfg.replace("    train_batch_size: 16\n    num_workers: 3\n", f"    train_batch_size: 32\n    num_workers: {workers}\n")
cfg = cfg.replace("  max_steps: 8000\n  accelerator: cpu\n  devices: 1\n  precision: 32\n",
                  f"  max_steps: 20000\n  max_time: \"{max_time}\"\n  accelerator: cpu\n  devices: 1\n  precision: bf16-mixed\n")
cfg = cfg.replace("panda_part_flip", dataset)
assert "bf16-mixed" in cfg and f"num_workers: {workers}" in cfg and max_time in cfg
train = r'''#!/bin/bash
set -x
export HOME=/home/ubuntu PATH=/home/ubuntu/.local/bin:$PATH
cd /home/ubuntu
say() { echo "CELL-TRAIN $(date -u +%H:%M:%S) $*" | sudo tee /dev/console; echo "$(date -u +%H:%M:%S) $*" > status.txt; curl -s -X PUT -T status.txt "$STATUS_URL" -o /dev/null; }
say "booted: $(nproc) vCPU, $(grep -o -m1 -E 'amx_bf16' /proc/cpuinfo || echo no-amx), $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone -q --filter=blob:none https://github.com/open-edge-platform/physical-ai-studio.git && git -C physical-ai-studio checkout -q 894ea9fb
uv venv -q -p 3.12 train
uv pip install -q -p train/bin/python --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch torchvision
uv pip install -q -p train/bin/python -e physical-ai-studio/library
train/bin/python -c "import torch, physicalai.policies; print('torch', torch.__version__, 'onednn', torch.backends.mkldnn.is_available())" > env.txt 2>&1
say "env ready: $(tail -1 env.txt)"
until curl -sf -o dataset.tgz "$DATA_URL"; do sleep 30; done
mkdir -p datasets && tar xzf dataset.tgz -C datasets
say "dataset: $(du -sh datasets | cut -f1), $(cat datasets/__DATASET__/meta/info.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("total_episodes"), "episodes", d.get("total_frames"), "frames")')"
cat > act.yaml <<'YAML'
__CFG__
YAML
(while sleep 120; do say "training: $(tail -1 runs/act/lightning_logs/*/metrics.csv 2>/dev/null | cut -c1-120) | ckpts: $(ls runs/act/lightning_logs/*/checkpoints 2>/dev/null | tr '\n' ' ')"; done) &
LOOP=$!
train/bin/physicalai fit --config act.yaml > fit.log 2>&1
echo "fit exit $?" >> fit.log
kill $LOOP
ckpt=$(ls -t runs/act/lightning_logs/*/checkpoints/*.ckpt | head -1)
say "uploading $ckpt"
curl -s -X PUT -T "$ckpt" "$CKPT_URL" -o /dev/null
tr '\r' '\n' < fit.log | grep -vE "^\s*$" | tail -400 > fit_tail.log
tar czf logs.tgz runs/act/lightning_logs fit_tail.log env.txt act.yaml
curl -s -X PUT -T logs.tgz "$LOGS_URL" -o /dev/null
say "done: $(tail -1 fit.log)"
sudo shutdown -h now
'''.replace("__CFG__", cfg).replace("__DATASET__", dataset)
exports = "".join(f"export {k}='{v}'\n" for k, v in urls.items())
ud = "#!/bin/bash\nexec > /var/log/cell-train.log 2>&1\nset -x\napt-get update -q && apt-get install -y -q git curl libgl1 libglib2.0-0\n" + \
     "cat > /home/ubuntu/train.sh <<'TRAINEOF'\n" + train + "TRAINEOF\n" + \
     "cat > /home/ubuntu/env.sh <<'ENVEOF'\n" + exports + "ENVEOF\n" + \
     "chown ubuntu:ubuntu /home/ubuntu/train.sh /home/ubuntu/env.sh; chmod +x /home/ubuntu/train.sh\n" + \
     "su ubuntu -c 'source /home/ubuntu/env.sh && /home/ubuntu/train.sh > /home/ubuntu/train.log 2>&1'\n"
open(f"{S}/user-data-{prefix}.sh", "w").write(ud)
print("user-data bytes", len(ud))
