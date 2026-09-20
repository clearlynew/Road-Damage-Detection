"""
extract_embedding.py -- real frozen-feature-extractor module for the ablation
track (spark_pipeline_ablation.py), replacing stub_embedding.py.

Deliberately reuses model_api.preprocess (the SAME shared preprocessing
module used by the fine-tuned track and by training) rather than redefining
its own transform. Both tracks must see identical resize/normalization or
the fine-tuned-vs-frozen+RF comparison isn't apples-to-apples.
"""
import io
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from model_api import preprocess  # SAME transform as the fine-tuned track -- do not redefine

EMBEDDING_DIM = 1280  # MobileNetV2's pooled output dim


def load_model():
    """
    Frozen ImageNet MobileNetV2 with its classifier head removed -- pure
    feature extractor, no fine-tuning. Call this ONCE per Spark partition
    (same load-once pattern as the primary track), never per row.
    """
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier = nn.Identity()
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def extract_embedding(image_bytes: bytes, model) -> "np.ndarray":
    """
    Takes raw patch bytes + a loaded frozen model, returns a 1280-dim
    embedding vector. Signature matches the stub's contract
    (image_bytes, model) -> vector, so spark_pipeline_ablation.py's calling
    code doesn't need to change shape.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with torch.no_grad():
        x = preprocess(img).unsqueeze(0)
        vec = model(x)[0]  # (1280,) -- classifier replaced with Identity above
    return vec.cpu().numpy()


if __name__ == "__main__":
    # Smoke test: does the frozen extractor run end-to-end and produce a
    # non-degenerate embedding?
    import numpy as np

    print("[test] loading frozen MobileNetV2 (ImageNet weights, no classifier head)")
    model = load_model()

    fake_img = Image.fromarray((np.random.rand(300, 300, 3) * 255).astype("uint8"))
    buf = io.BytesIO()
    fake_img.save(buf, format="JPEG")

    vec = extract_embedding(buf.getvalue(), model)
    print(f"[test] embedding shape: {vec.shape}, dtype: {vec.dtype}")
    assert vec.shape == (EMBEDDING_DIM,), f"expected shape ({EMBEDDING_DIM},), got {vec.shape}"
    assert not np.isnan(vec).any(), "NaN values in embedding!"
    print("[test] PASS: frozen extractor produces a well-shaped, non-NaN embedding")