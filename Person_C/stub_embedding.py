"""
STUB for Person 1's deliverable.

Real contract (per proposal Sec 5 / handoff note):
    extract_embedding(image) -> vector

Person 1 will hand you a real version of this file (or just this function)
that loads a frozen MobileNetV2 and returns a 1280-dim embedding.
When that lands: replace this file's extract_embedding with theirs.
The rest of your pipeline should not need to change AT ALL if the
signature stays (image_bytes_or_array) -> 1D vector.

This stub:
  - fakes "loading a model" with a fixed cost, so you can test the
    "model loaded once per partition" pattern (Sec 6, decision #2)
  - returns a deterministic, reproducible fake embedding (seeded by
    a hash of the image bytes) so re-runs are stable for testing
  - matches MobileNetV2's real output dim (1280) so downstream
    Parquet schemas / MLlib code won't need to change later
"""
import hashlib
import time
import numpy as np

EMBEDDING_DIM = 1280  # MobileNetV2 pooled output dim; change to 512 for ResNet18


class StubModel:
    """Mimics a loaded CNN backbone. Instantiate ONCE per partition."""

    def __init__(self, load_delay_sec: float = 0.05):
        # simulate the real cost of loading torchvision weights,
        # so your "load once per partition" test is meaningful
        time.sleep(load_delay_sec)
        self.loaded = True
        self.load_count_marker = time.time()  # useful to prove it's not reloaded per-row

    def embed(self, image_bytes: bytes) -> np.ndarray:
        # deterministic pseudo-embedding derived from the image content
        h = hashlib.sha256(image_bytes).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.RandomState(seed)
        return rng.rand(EMBEDDING_DIM).astype(np.float32)


def extract_embedding(image_bytes: bytes, model: StubModel) -> np.ndarray:
    """
    Same call signature you'll use with the real function, except the
    real one from Person 1 will likely be:
        extract_embedding(image_bytes) -> vector
    with the model loaded internally via a module-level cache, OR
    you'll pass the model in explicitly (confirm which with Person 1 --
    it changes how you write the mapPartitions wrapper below).
    """
    return model.embed(image_bytes)