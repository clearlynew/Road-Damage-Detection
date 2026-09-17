"""
STUB for Person 2's deliverable.

Real contract (per proposal Sec 5 / Sec 2):
  - raw JPEGs organized by country, in HDFS
  - a parsed label table (majority damage class per image) as Parquet in HDFS

This script fakes both so you can build/test your Spark job end-to-end.
Swap this out for reads against real HDFS paths once Person 2 delivers.

Simulates HDFS locally under ./hdfs_sim/ since real HDFS isn't running here.
Replace local paths with hdfs:// URIs (or your team's HDFS mount) later --
that's the ONLY thing that should need to change in spark_pipeline.py.
"""
import os
import random
import pandas as pd
from PIL import Image
import numpy as np

HDFS_SIM_ROOT = "hdfs_sim"
COUNTRIES = ["Japan", "India", "US", "CzechRepublic", "China"]  # 5-country run one, per Sec 2
DAMAGE_CLASSES = ["D00", "D10", "D20", "D40"]
N_IMAGES_PER_COUNTRY = 40  # small fake set; bump up to stress-test partitioning


def make_fake_image(path, size=(224, 224)):
    arr = (np.random.rand(*size, 3) * 255).astype("uint8")
    Image.fromarray(arr).save(path, format="JPEG")


def generate():
    raw_dir = os.path.join(HDFS_SIM_ROOT, "raw_images")
    label_dir = os.path.join(HDFS_SIM_ROOT, "label_table")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    rows = []
    for country in COUNTRIES:
        country_dir = os.path.join(raw_dir, country)
        os.makedirs(country_dir, exist_ok=True)
        for i in range(N_IMAGES_PER_COUNTRY):
            fname = f"{country}_{i:05d}.jpg"
            fpath = os.path.join(country_dir, fname)
            make_fake_image(fpath)
            rows.append({
                "image_path": fpath,
                "country": country,
                "label": random.choice(DAMAGE_CLASSES),  # stand-in for majority-class label
            })

    # a few "dropped" no-label images, matching Sec 2's dropped-count logging behavior
    n_dropped = 3
    print(f"[stub] simulating {n_dropped} unlabeled images dropped (per Sec 2 label scheme)")

    label_df = pd.DataFrame(rows)
    label_df.to_parquet(os.path.join(label_dir, "labels.parquet"), index=False)
    print(f"[stub] wrote {len(label_df)} labeled rows across {len(COUNTRIES)} countries")
    print(f"[stub] raw images under {raw_dir}/<country>/")
    print(f"[stub] label table at {label_dir}/labels.parquet")


if __name__ == "__main__":
    generate()