"""
model_api.py -- shared contract between Colab (training) and Spark (inference),
per proposal v4, Section 4 Stage 1/2 and Section 6 decision 2/3.

CRITICAL (v4 decision 2): this exact file must be used UNCHANGED by both the
Colab training script and this Spark pipeline. Any difference in preprocessing
between training and inference is treated as a bug (train/inference skew),
not a tuning knob. Do not let Person 1 have her own copy that drifts from this.
"""
import torch
from torchvision import models, transforms

CLASSES = ["D00", "D10", "D20", "D40", "background"]  # MUST match sorted() order --
# torchvision's ImageFolder (used in training) assigns class indices by
# alphabetically sorting subfolder names. If this list isn't in that same
# order, the trained model's output index N means a DIFFERENT class during
# training than what predict_patch() below assumes during inference -- a
# silent, catastrophic train/inference skew (the exact risk v4 Section 8
# calls out). Verified: sorted(["background","D00","D10","D20","D40"]) ==
# ["D00","D10","D20","D40","background"] because uppercase letters sort
# before lowercase in ASCII. Do not reorder this list without also checking
# train_finetune.py's assertion still passes.
NUM_CLASSES = len(CLASSES)

# Shared preprocessing -- used identically in training (Colab) and inference (Spark).
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(num_classes: int = NUM_CLASSES):
    """Architecture only, no weights. Used by both training (before fine-tuning)
    and inference (before loading trained weights)."""
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = torch.nn.Linear(m.last_channel, num_classes)
    return m


def load_model(weights_path: str):
    """
    Load a TRAINED model from a .pt file (local path or a path already copied
    from HDFS -- see note in spark_pipeline.py's run_inference()).
    Call this ONCE per Spark partition (v4 decision 3, same load-once pattern
    as v3's decision 2) -- never per row.
    """
    m = build_model()
    state_dict = torch.load(weights_path, map_location="cpu")
    m.load_state_dict(state_dict)
    m.eval()
    return m


def predict_patch(image_bytes: bytes, model) -> dict:
    """
    NEW v4 contract (was extract_embedding in v3). Takes raw image/patch bytes
    and a loaded model, returns class probabilities directly -- no separate
    MLlib classifier needed for this track (that's the ablation track's job).
    """
    import io
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with torch.no_grad():
        x = preprocess(img).unsqueeze(0)
        logits = model(x)[0]
        probs = torch.softmax(logits, dim=0)
    return {c: float(p) for c, p in zip(CLASSES, probs)}


if __name__ == "__main__":
    # Smoke test: does the untrained architecture at least run end-to-end?
    # (Can't test real trained weights here -- Person 1 hasn't delivered the
    # .pt yet. This only proves the code path/shapes are correct.)
    import io
    import numpy as np
    from PIL import Image

    print("[test] building untrained model (architecture-only smoke test)")
    model = build_model()
    model.eval()

    fake_img = Image.fromarray((np.random.rand(300, 300, 3) * 255).astype("uint8"))
    buf = io.BytesIO()
    fake_img.save(buf, format="JPEG")

    probs = predict_patch(buf.getvalue(), model)
    print("[test] predict_patch output:", probs)
    assert set(probs.keys()) == set(CLASSES), "class set mismatch!"
    assert abs(sum(probs.values()) - 1.0) < 1e-4, "probabilities don't sum to 1!"
    print("[test] PASS: contract shape is correct (untrained weights, so values are meaningless)")