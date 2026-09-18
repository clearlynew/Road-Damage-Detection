"""
ABLATION TRACK (v4 Section 5, v3 pipeline retained verbatim). Frozen MobileNetV2
embeddings -> Spark MLlib RandomForest/GBT. This is the comparison model against
the fine-tuned track in spark_pipeline.py, and the fallback if fine-tuning or
Colab fails (v4 decision 4 / Section 8 risk).
"""
import io
import os
import time

# Fix for Windows machines with hostnames Spark's URL parser can't handle
# (e.g. hostnames starting with a dash). Must be set BEFORE SparkSession is built.
os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

# Windows only: Spark needs winutils.exe to write files, even for purely
# local runs. If HADOOP_HOME isn't already set system-wide, point it at
# wherever you extracted winutils.exe (see README for download link).
# Harmless no-op on Linux/Mac.
if os.name == "nt" and "HADOOP_HOME" not in os.environ:
    os.environ["HADOOP_HOME"] = r"C:\hadoop"  # <-- change if you put it elsewhere
    os.environ["PATH"] = os.environ["HADOOP_HOME"] + r"\bin;" + os.environ["PATH"]

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

# SWAP 1: real embedding module (was: stub_embedding)
from extract_embedding import load_model, extract_embedding, EMBEDDING_DIM

# SWAP 2: now pointing at real HDFS instead of local hdfs_sim/
LABEL_TABLE_PATH = "hdfs://localhost:9000/rdd2022/label_table/labels.parquet"
RAW_IMAGES_PATH = "hdfs://localhost:9000/rdd2022/raw_images/*/*.jpg"  # all countries, all images
EMBEDDINGS_OUT_PATH = "hdfs://localhost:9000/rdd2022/embeddings"


def process_partition(rows):
    """
    Runs once per Spark partition. This is where Sec 6 decision #2 lives:
    the model must be instantiated ONCE here, not once per row.

    Each row now carries image bytes directly (already read from HDFS via
    Spark's binaryFile source in `run()`), so there's no local open() call
    here -- that's what breaks once paths point into HDFS instead of disk.
    """
    model = load_model()  # <-- loaded once per partition
    load_marker = id(model)  # proxy for "which model instance" -- same value for every row in this partition proves load-once

    results = []
    for row in rows:
        vec = extract_embedding(row["content"], model)
        results.append({
            "image_path": row["path"],
            "country": row["country"],
            "label": row["label"],
            "embedding": vec.tolist(),
            "partition_model_load_marker": float(load_marker % 1_000_000),  # truncated for the FloatType schema
        })
    return iter(results)


def check_embedding_sanity(result_df, sample_size: int = 50):
    """
    Catches a broken/degenerate embedding stage BEFORE it wastes a full
    classifier + mAP run downstream. A real (even weak) frozen-backbone
    embedding should have per-dimension variance and should NOT be identical
    across different input images. If this check fails, the bug is upstream
    of the classifier -- don't blame "low accuracy" on the model choice yet.
    """
    import numpy as np

    sample = result_df.select("embedding", "label").limit(sample_size).collect()
    if len(sample) < 2:
        print("[sanity] not enough rows to check (need >= 2)")
        return

    vectors = np.array([row["embedding"] for row in sample])

    n_nan = np.isnan(vectors).sum()
    n_inf = np.isinf(vectors).sum()
    per_dim_std = vectors.std(axis=0)
    n_dead_dims = (per_dim_std < 1e-6).sum()  # dimensions that never vary -- suspicious
    pairwise_identical = 0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if np.allclose(vectors[i], vectors[j], atol=1e-6):
                pairwise_identical += 1

    print("\n[sanity] --- embedding sanity check (n={}) ---".format(len(sample)))
    print(f"[sanity] NaN values: {n_nan} (should be 0)")
    print(f"[sanity] Inf values: {n_inf} (should be 0)")
    print(f"[sanity] dead dimensions (zero variance across sample): {n_dead_dims} / {vectors.shape[1]} "
          f"(a few is normal for frozen features; most/all dead = broken forward pass)")
    print(f"[sanity] identical embedding pairs across DIFFERENT images: {pairwise_identical} "
          f"(should be 0 or near-0 -- if high, every image is producing the same vector, "
          f"which means the model isn't actually looking at the image)")
    print(f"[sanity] value range: min={vectors.min():.4f} max={vectors.max():.4f} mean={vectors.mean():.4f}")

    if n_nan > 0 or n_inf > 0:
        print("[sanity] FAIL: NaN/Inf in embeddings -- fix the forward pass before doing anything else")
    elif n_dead_dims > vectors.shape[1] * 0.5:
        print("[sanity] WARNING: over half the dimensions are dead -- check preprocessing/normalization")
    elif pairwise_identical > 0:
        print("[sanity] WARNING: some images produced identical embeddings -- check image reading/decoding")
    else:
        print("[sanity] PASS: embeddings look non-degenerate. Low downstream accuracy, if it happens, "
              "is then a real (if weak) signal -- not a broken pipeline.")
    print("[sanity] --- end check ---\n")


def run(num_partitions: int = 4, verbose: bool = True):
    spark = (
        SparkSession.builder
        .appName("RDD2022-EmbeddingExtraction")
        .master(f"local[{num_partitions}]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Read raw image BYTES directly from HDFS -- this is what replaces the
    # old local open(row["image_path"], "rb") call, which can't read hdfs:// paths.
    images_df = (
        spark.read.format("binaryFile")
        .load(RAW_IMAGES_PATH)
        .selectExpr("path", "content")
    )

    # image_path in the label table (written by generate_stub_data.py) was a
    # LOCAL path like "hdfs_sim/raw_images/Japan/Japan_00001.jpg". Join on the
    # filename only, since the HDFS path prefix will differ.
    from pyspark.sql.functions import element_at, split

    images_df = images_df.withColumn("filename", element_at(split("path", "/"), -1))

    label_df = spark.read.parquet(LABEL_TABLE_PATH)
    label_df = label_df.withColumn("filename", element_at(split("image_path", "/"), -1))

    joined_df = images_df.join(label_df, on="filename", how="inner").select(
        "path", "content", "country", "label"
    )
    joined_df = joined_df.repartition(num_partitions)

    t0 = time.time()
    result_rdd = joined_df.rdd.mapPartitions(lambda rows: process_partition([r.asDict() for r in rows]))

    schema = StructType([
        StructField("image_path", StringType()),
        StructField("country", StringType()),
        StructField("label", StringType()),
        StructField("embedding", ArrayType(FloatType())),
        StructField("partition_model_load_marker", FloatType()),
    ])
    result_df = spark.createDataFrame(result_rdd, schema=schema)
    result_df.write.mode("overwrite").parquet(EMBEDDINGS_OUT_PATH)
    elapsed = time.time() - t0

    n_rows = result_df.count()
    n_distinct_loads = result_df.select("partition_model_load_marker").distinct().count()

    if verbose:
        print(f"[pipeline] partitions={num_partitions} rows={n_rows} elapsed={elapsed:.2f}s "
              f"throughput={n_rows/elapsed:.1f} img/s")
        print(f"[pipeline] distinct model-load markers = {n_distinct_loads} "
              f"(should be <= num_partitions={num_partitions} -- proves load-once-per-partition)")
        check_embedding_sanity(result_df)

    spark.stop()
    return {"partitions": num_partitions, "rows": n_rows, "elapsed_sec": elapsed}


if __name__ == "__main__":
    # This is your Sec 6 decision #2 pilot test: run small first (10-50 images)
    # before any full run. Stub data is 200 rows -- fine for this smoke test.
    run(num_partitions=4)