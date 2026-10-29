# The whole cell in one CPU-only image: Intel's MuJoCo simulator (physicalai + this repo's plugin patch),
# the OpenVINO perception service (Grounding DINO + SAM-6D), the console / cell API, and the ACT policy
# runtime. Training stays outside (AWS c7i scripts or the physical-ai-studio env).
#
#   docker build -f docker/Dockerfile -t intel-manip-cell .
#   docker compose -f docker/compose.yaml up
#
# Model weights are not baked in; mount them (see docker/README.md).

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ARG PHYSICALAI_COMMIT=ec92b29
ARG SAM6D_COMMIT=89d39f8
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    SAM6D_ROOT=/opt/OpenVINO-SAM-6D/SAM-6D

# Optional extra CAs for builds behind a TLS-inspecting proxy (docker/certs/*.crt; none by default)
COPY docker/certs/ /usr/local/share/ca-certificates/extra/
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential cmake libeigen3-dev libgomp1 \
        libosmesa6 libgl1 libegl1 libglib2.0-0 ffmpeg fonts-dejavu-core \
    && update-ca-certificates && rm -rf /var/lib/apt/lists/*
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt UV_SYSTEM_CERTS=1
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

COPY patches /opt/patches

# Intel physicalai (MuJoCo plugin, SharedRobot, RobotRuntime, InferenceModel) + RGB-D / timing patch
RUN git clone https://github.com/openvinotoolkit/physicalai.git /opt/physicalai \
    && git -C /opt/physicalai checkout ${PHYSICALAI_COMMIT} \
    && git -C /opt/physicalai apply /opt/patches/physicalai-mujoco-plugin.patch
COPY docker/requirements-cell.txt /opt/requirements-cell.txt
RUN uv venv -p 3.13 /opt/venv/cell \
    && uv pip install -p /opt/venv/cell -r /opt/requirements-cell.txt \
    && uv pip install -p /opt/venv/cell --no-deps -e /opt/physicalai \
         -e /opt/physicalai/packages/physicalai-mujoco-plugin -e /opt/physicalai/packages/physicalai-studio-plugin

# Perception: OpenVINO 2026.4.1, CPU torch (DINO post-processing only), OpenVINO-SAM-6D + pointnet2 ops
COPY docker/requirements-perception.txt /opt/requirements-perception.txt
RUN uv venv -p 3.11 /opt/venv/perception \
    && uv pip install -p /opt/venv/perception torch==2.5.0 torchvision==0.20.0 \
         --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install -p /opt/venv/perception -r /opt/requirements-perception.txt \
    && uv pip install -p /opt/venv/perception --no-deps pyrender==0.1.45  # pins PyOpenGL 3.1.0; 3.1.7 works with OSMesa
RUN git clone https://github.com/18582088138/OpenVINO-SAM-6D.git /opt/OpenVINO-SAM-6D \
    && git -C /opt/OpenVINO-SAM-6D checkout ${SAM6D_COMMIT} \
    && git -C /opt/OpenVINO-SAM-6D apply /opt/patches/openvino-sam6d-register-pointnet2-ops.patch \
    && cd ${SAM6D_ROOT}/Pose_Estimation_Model/model/ov_pointnet2_op \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
         -DPython3_EXECUTABLE=/opt/venv/perception/bin/python \
    && cmake --build build -j2 \
    && rm -rf ${SAM6D_ROOT}/Data ${SAM6D_ROOT}/Pose_Estimation_Model/checkpoints

WORKDIR /app
COPY scenes scenes
COPY console console
COPY scripts scripts
COPY v2 v2
COPY v3 v3
COPY docker/start.sh docker/start.sh

EXPOSE 8000 8080 9090 8100
CMD ["bash", "docker/start.sh"]
