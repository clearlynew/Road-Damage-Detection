"""
extract_embedding.py — cleaned version of Person 1's rd2022_training.py

WHAT WAS REMOVED from the original script, and why:
  - YOLOv8 detection training/inference: out of scope (proposal is image
    classification, not object detection)
  - Fine-tuning of the last 3 MobileNetV2 blocks
    (`for param in model.features[-3:].parameters(): param.requires_grad = True`):
    violates proposal Sec 6 decision #1 ("frozen") and Sec 5's simplicity rule
    ("no fine-tuning")
  - The classifier head (`model.classifier = nn.Sequential(...)`) and the
    whole training loop: Person 1 hands off FEATURES, not a trained classifier
    -- classification is Person 4's job (Spark MLlib), per Sec 5
  - 5-class label handling: proposal specifies 4 classes (D00,D10,D20,D40),
    not 5 (D00,D10,D20,D40,D43)

WHAT WAS KEPT, unchanged in substance:
  - The image transform (resize 224x224, ImageNet normalization) -- this
    matches proposal Sec 2 exactly
  - Loading MobileNetV2 with ImageNet-pretrained weights via torchvision

CONTRACT this file provides (matches the proposal's handoff spec exactly):
    extract_embedding(image) -> vector
"""
import io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 1280  # MobileNetV2's pooled feature dimension

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def load_model():
    """
    Load MobileNetV2 with ImageNet weights, FROZEN, no classifier head.
    Call this ONCE per Spark partition (per proposal Sec 6 decision #2) --
    not once per image, which is the "biggest known pitfall" the proposal
    calls out.
    """
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()  # drop classifier head -- we want features, not logits
    for param in model.parameters():
        param.requires_grad = False  # frozen, per Sec 6 decision #1 -- no fine-tuning anywhere
    model.eval()
    model = model.to(DEVICE)
    return model


def extract_embedding(image_bytes: bytes, model) -> np.ndarray:
    """
    Contract matches proposal's handoff spec: extract_embedding(image) -> vector.
    `model` is loaded once via load_model() and passed in (or cached globally --
    confirm with Person 3 which pattern their mapPartitions wrapper expects).

    Input: raw image bytes (e.g. read from HDFS via Spark's binaryFile source).
    Output: 1280-dim numpy float32 vector, no gradient tracking, CPU-side.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model.features(tensor)              # conv feature maps
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1)  # global avg pool
        vec = torch.flatten(pooled, 1)                  # (1, 1280)

    return vec.squeeze(0).cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    # Smoke test: does this even run and produce the right shape?
    # Uses a synthetic image since no real RDD2022 data is available here.
    import time

    print(f"[test] device = {DEVICE}")

    t0 = time.time()
    model = load_model()
    print(f"[test] model loaded in {time.time()-t0:.2f}s")

    # fake a JPEG in memory
    fake_img = Image.fromarray((np.random.rand(300, 300, 3) * 255).astype("uint8"))
    buf = io.BytesIO()
    fake_img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    t0 = time.time()
    vec = extract_embedding(image_bytes, model)
    print(f"[test] extraction took {time.time()-t0:.3f}s")
    print(f"[test] output shape: {vec.shape}, dtype: {vec.dtype}")
    print(f"[test] matches expected EMBEDDING_DIM={EMBEDDING_DIM}: {vec.shape[0] == EMBEDDING_DIM}")
    print(f"[test] sample values: {vec[:5]}")

    # confirm frozen: no params should require grad
    n_trainable = sum(p.requires_grad for p in model.parameters())
    print(f"[test] trainable params (should be 0): {n_trainable}")