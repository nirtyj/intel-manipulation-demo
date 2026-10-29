import os, boto3
S = os.path.dirname(os.path.abspath(__file__))
b = os.environ.get("TTC_BUCKET", "talk-to-the-cell-688567285324")
s3 = boto3.Session(profile_name="nirty", region_name="us-east-1").client("s3")
ttl = 12 * 3600
get = lambda k: s3.generate_presigned_url("get_object", Params={"Bucket": b, "Key": k}, ExpiresIn=ttl)
put = lambda k: s3.generate_presigned_url("put_object", Params={"Bucket": b, "Key": k}, ExpiresIn=ttl)
env = {"CODE_URL": get("demo/cell-code.tgz"), "MODELS_URL": get("models/cell-models.tar"), "STATUS_URL": put("demo/status.txt")}
script = r'''#!/bin/bash
exec > /var/log/cell-demo.log 2>&1
set -x
''' + "".join(f"export {k}='{v}'\n" for k, v in env.items()) + r'''
say() { echo "$(date -u +%H:%M:%S) $*" > /tmp/status.txt; curl -s -X PUT -T /tmp/status.txt "$STATUS_URL" -o /dev/null; }
say "booted: $(nproc) vCPU, $(free -g | awk '/Mem/ {print $2}') GB"
apt-get update -q && apt-get install -y -q docker.io docker-compose-v2 curl
mkdir -p /opt/cell && cd /opt/cell
curl -sf -o code.tgz "$CODE_URL" && tar xzf code.tgz && rm code.tgz
curl -sf -o models.tar "$MODELS_URL" && tar xf models.tar && rm models.tar
mkdir -p runs && chown -R ubuntu:ubuntu /opt/cell
say "building the image (about 20-30 min)"
if ! docker build --build-arg BASE_IMAGE=mirror.gcr.io/library/ubuntu:24.04 -f docker/Dockerfile -t intel-manip-cell . > /var/log/cell-build.log 2>&1; then
  say "build failed: $(tail -3 /var/log/cell-build.log | tr '\n' ' ')"; exit 1; fi
say "starting the cell"
docker compose -f docker/compose.yaml up -d
for i in $(seq 1 120); do curl -sf localhost:8100/health | grep -q '"ready":true' && break; sleep 5; done
curl -sf localhost:8000/ -o /dev/null && say "ready: console on :8000, viewer on :9090" || say "console not answering: $(docker compose -f docker/compose.yaml logs --tail 5 | tr '\n' ' ')"
'''
open(f"{S}/user-data-demo.sh", "w").write(script)
print("user-data bytes", len(script))
