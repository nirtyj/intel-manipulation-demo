"""Convert Grounding DINO (tiny) from Hugging Face to OpenVINO IR and check it against PyTorch.

    python -m v2.perception.convert_dino --out models/grounding_dino_tiny.xml
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import openvino as ov
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def _logit(x: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    """torch.special.logit in basic ops, for OpenVINO releases without a rule for aten::special_logit."""
    if eps is not None:
        x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


if os.environ.get("DINO_LOGIT_PATCH", "1") == "1":
    torch.special.logit = _logit  # Grounding DINO's decoder calls it for its reference points
INPUTS = ["pixel_values", "input_ids", "token_type_ids", "attention_mask", "pixel_mask"]


class Wrapped(torch.nn.Module):
    """Positional inputs and plain tensor outputs (logits, boxes) for tracing."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_ids, token_type_ids, attention_mask, pixel_mask):
        out = self.model(pixel_values=pixel_values, input_ids=input_ids, token_type_ids=token_type_ids,
                         attention_mask=attention_mask, pixel_mask=pixel_mask, return_dict=True)
        return out.logits, out.pred_boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/grounding_dino_tiny.xml")
    ap.add_argument("--image", default="/home/user/OpenVINO-SAM-6D/SAM-6D/Data/Example/rgb.png")
    ap.add_argument("--prompt", default="watering can.")
    ap.add_argument("--fp16", action="store_true", help="compress weights to FP16")
    a = ap.parse_args()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).eval()
    wrapped = Wrapped(model).eval()
    inputs = proc(images=Image.open(a.image).convert("RGB"), text=a.prompt, return_tensors="pt")
    example = tuple(inputs[k] for k in INPUTS)
    t0 = time.time()
    with torch.no_grad():
        ovm = ov.convert_model(wrapped, example_input=example)
    for i, name in enumerate(INPUTS):
        ovm.inputs[i].get_tensor().set_names({name})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    ov.save_model(ovm, a.out, compress_to_fp16=a.fp16)
    print(f"converted and saved {a.out} in {time.time() - t0:.0f} s", flush=True)
    compiled = ov.Core().compile_model(a.out, "CPU")
    feed = {k: inputs[k].numpy() for k in INPUTS}
    compiled(feed)
    t0 = time.time()
    res = compiled(feed)
    dt = time.time() - t0
    with torch.no_grad():
        ref = wrapped(*example)
    print(f"OpenVINO CPU {dt * 1000:.0f} ms | max |logit diff| {np.abs(res[0] - ref[0].numpy()).max():.3f} "
          f"| max |box diff| {np.abs(res[1] - ref[1].numpy()).max():.4f}", flush=True)


if __name__ == "__main__":
    main()
