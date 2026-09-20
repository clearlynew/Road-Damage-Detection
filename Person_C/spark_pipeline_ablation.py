"""
ABLATION TRACK (v4 Section 5, v3 pipeline retained + extended for real data).
Distributed SLIDING-WINDOW PATCH EXTRACTION + frozen MobileNetV2 embeddings.
This is the comparison-model data source for the frozen+RandomForest track,
and the fallback if fine-tuning or Colab fails (v4 decision 4 / Section 8 risk).

FIX vs. the previous version: the old code extracted ONE embedding per whole
image (no sliding window) and joined against a stub schema (image_path,
country, label). This version extracts a real 224x224 sliding-window patch
grid per image (identical geometry to spark_pipeline.py and Person 2's
patch-dataset-build notebook) and joins against the REAL ground-truth schema
(image_path, country, boxes -- a list of per-image damage boxes), emitting
one embedding row PER PATCH with patch coordinates.

Downstream: a separate Spark MLlib step (not this file) fits
RandomForestClassifier/GBTClassifier on the embeddings this script writes.
That MLlib training step is not yet included here -- flag if you want it
written next.
"""
import io
import os
import time

os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

if os.name == "nt" and "HADOOP_HOME" not in os.environ:
    os.environ["HADOOP_HOME"] = r"C:\hadoop"
    os.environ["PATH"] = os.environ["HADOOP_HOME"] + r"\bin;" + os.environ["PATH"]

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType, FloatType, IntegerType
)

from PIL import Image

# Real embedding module (was: stub_embedding)
from extract_embedding import load_model, extract_embedding, EMBEDDING_DIM

# -----------------------------------------------------------------------------
# SHARED GEOMETRY CONSTANTS -- MUST exactly match spark_pipeline.py and
# Person 2's patch-dataset-build notebook. Both tracks must see identical
# patch geometry or the fine-tuned vs. frozen+RF comparison isn't apples-to-
# apples.
# -----------------------------------------------------------------------------
RESIZE_LONG_SIDE = 1024
PATCH_SIZE = 224
STRIDE = 112

LABEL_TABLE_PATH = "hdfs://localhost:9000/rdd2022/ground_truth/original_images.parquet"
RAW_IMAGES_PATH = "hdfs://localhost:9000/rdd2022/raw_images_eval/*/*.jpg"
EMBEDDINGS_OUT_PATH = "hdfs://localhost:9000/rdd2022/embeddings"


def get_resize_scale(width, height, max_long_side=RESIZE_LONG_SIDE):
    long_side = max(width, height)
    return 1.0 if long_side <= max_long_side or long_side == 0 else max_long_side / long_side


def sliding_window_coords(width, height, patch_size=PATCH_SIZE, stride=STRIDE):
    if width < patch_size or height < patch_size:
        return
    xs = list(range(0, width - patch_size + 1, stride))
    if xs[-1] != width - patch_size:
        xs.append(width - patch_size)
    ys = list(range(0, height - patch_size + 1, stride))
    if ys[-1] != height - patch_size:
        ys.append(height - patch_size)
    for y0 in ys:
        for x0 in xs:
            yield x0, y0, x0 + patch_size, y0 + patch_size


def process_partition(rows):
    """
    Runs once per Spark partition. Model instantiated ONCE here (Sec 6
    decision #2), then for each image: resize -> slide window -> crop ->
    extract_embedding per crop. Yields one row per patch.
    """
    model = load_model()  # <-- loaded once per partition
    load_marker = id(model)

    results = []
    for row in rows:
        try:
            img = Image.open(io.BytesIO(row["content"])).convert("RGB")
        except Exception as e:
            print(f"[ablation] skipping unreadable image {row['path']}: {e}")
            continue

        width, height = img.size
        scale = get_resize_scale(width, height)
        if scale != 1.0:
            img = img.resize(
                (int(round(width * scale)), int(round(height * scale))), Image.BILINEAR
            )
        resized_w, resized_h = img.size

        for (x0, y0, x1, y1) in sliding_window_coords(resized_w, resized_h):
            crop = img.crop((x0, y0, x1, y1))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG")
            vec = extract_embedding(buf.getvalue(), model)

            results.append({
                "image_path": row["path"],
                "country": row["country"],
                "scale": float(scale),
                "patch_x0": x0, "patch_y0": y0, "patch_x1": x1, "patch_y1": y1,
                "embedding": vec.tolist(),
                "partition_model_load_marker": float(load_marker % 1_000_000),
            })
    return iter(results)


def check_embedding_sanity(result_df, sample_size: int = 50):
    """
    Catches a broken/degenerate embedding stage BEFORE it wastes a full
    classifier run downstream.
    """
    import numpy as np

    sample = result_df.select("embedding").limit(sample_size).collect()
    if len(sample) < 2:
        print("[sanity] not enough rows to check (need >= 2)")
        return

    vectors = np.array([row["embedding"] for row in sample])

    n_nan = np.isnan(vectors).sum()
    n_inf = np.isinf(vectors).sum()
    per_dim_std = vectors.std(axis=0)
    n_dead_dims = (per_dim_std < 1e-6).sum()
    pairwise_identical = 0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if np.allclose(vectors[i], vectors[j], atol=1e-6):
                pairwise_identical += 1

    print(f"\n[sanity] --- embedding sanity check (n={len(sample)}) ---")
    print(f"[sanity] NaN values: {n_nan} (should be 0)")
    print(f"[sanity] Inf values: {n_inf} (should be 0)")
    print(f"[sanity] dead dimensions: {n_dead_dims} / {vectors.shape[1]}")
    print(f"[sanity] identical embedding pairs across different patches: {pairwise_identical} (should be ~0)")
    print(f"[sanity] value range: min={vectors.min():.4f} max={vectors.max():.4f} mean={vectors.mean():.4f}")

    if n_nan > 0 or n_inf > 0:
        print("[sanity] FAIL: NaN/Inf in embeddings -- fix the forward pass before doing anything else")
    elif n_dead_dims > vectors.shape[1] * 0.5:
        print("[sanity] WARNING: over half the dimensions are dead -- check preprocessing/normalization")
    elif pairwise_identical > 0:
        print("[sanity] WARNING: some patches produced identical embeddings -- check image reading/decoding")
    else:
        print("[sanity] PASS: embeddings look non-degenerate.")
    print("[sanity] --- end check ---\n")


def run(num_partitions: int = 4, verbose: bool = True):
    spark = (
        SparkSession.builder
        .appName("RDD2022-EmbeddingExtraction")
        .master(f"local[{num_partitions}]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    images_df = (
        spark.read.format("binaryFile")
        .load(RAW_IMAGES_PATH)
        .selectExpr("path", "content")
    )

    from pyspark.sql.functions import element_at, split

    images_df = images_df.withColumn("filename", element_at(split("path", "/"), -1))

    label_df = spark.read.parquet(LABEL_TABLE_PATH)
    label_df = label_df.withColumn("filename", element_at(split("image_path", "/"), -1))

    joined_df = images_df.join(label_df, on="filename", how="inner").select(
        "path", "content", "country"   # boxes not needed for the embedding track itself
    )
    joined_df = joined_df.repartition(num_partitions)

    t0 = time.time()
    result_rdd = joined_df.rdd.mapPartitions(lambda rows: process_partition([r.asDict() for r in rows]))

    schema = StructType([
        StructField("image_path", StringType()),
        StructField("country", StringType()),
        StructField("scale", FloatType()),
        StructField("patch_x0", IntegerType()), StructField("patch_y0", IntegerType()),
        StructField("patch_x1", IntegerType()), StructField("patch_y1", IntegerType()),
        StructField("embedding", ArrayType(FloatType())),
        StructField("partition_model_load_marker", FloatType()),
    ])
    result_df = spark.createDataFrame(result_rdd, schema=schema)
    result_df.write.mode("overwrite").parquet(EMBEDDINGS_OUT_PATH)
    elapsed = time.time() - t0

    n_rows = result_df.count()
    n_distinct_loads = result_df.select("partition_model_load_marker").distinct().count()

    if verbose:
        print(f"[pipeline] partitions={num_partitions} patches={n_rows} elapsed={elapsed:.2f}s "
              f"throughput={n_rows/elapsed:.1f} patches/s")
        print(f"[pipeline] distinct model-load markers = {n_distinct_loads} "
              f"(should be <= num_partitions={num_partitions})")
        check_embedding_sanity(result_df)

    spark.stop()
    return {"partitions": num_partitions, "rows": n_rows, "elapsed_sec": elapsed}


if __name__ == "__main__":
    # Pilot test: run small first, before any full run.
    run(num_partitions=4)