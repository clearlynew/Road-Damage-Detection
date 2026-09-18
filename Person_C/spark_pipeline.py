"""
PRIMARY TRACK (v4 Section 4, Stage 2). Distributed inference with a
FINE-TUNED model, loaded once per partition from a .pt file in HDFS.

Fine-tuning itself (Stage 1) happens OFF this file, on a single machine
(Colab), per v4's "train once, distribute inference" design. This file only
runs inference -- it does NOT train anything.

The v3 frozen-embedding pipeline is preserved separately in
spark_pipeline_ablation.py for the MLlib ablation track (v4 keeps it verbatim).

Requires: Person 1's trained weights already uploaded to MODEL_WEIGHTS_HDFS_PATH.
This will fail loudly and clearly if that file doesn't exist yet -- that's
intentional, don't silently fall back to untrained weights.
"""
import os
import time

os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

if os.name == "nt" and "HADOOP_HOME" not in os.environ:
    os.environ["HADOOP_HOME"] = r"C:\hadoop"
    os.environ["PATH"] = os.environ["HADOOP_HOME"] + r"\bin;" + os.environ["PATH"]

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, MapType

from model_api import load_model, predict_patch, CLASSES

LABEL_TABLE_PATH = "hdfs://localhost:9000/rdd2022/label_table/labels.parquet"
RAW_IMAGES_PATH = "hdfs://localhost:9000/rdd2022/raw_images/*/*.jpg"
INFERENCE_OUT_PATH = "hdfs://localhost:9000/rdd2022/inference_probs"
MODEL_WEIGHTS_HDFS_PATH = "hdfs://localhost:9000/models/rdamage_mobilenetv2.pt"


def fetch_weights_to_local(spark, hdfs_path: str, local_path: str = "/tmp/model_weights.pt"):
    """
    torch.load() needs a local file path, not an hdfs:// URI. This copies the
    .pt from HDFS to local disk using Spark's own Hadoop filesystem client
    (via py4j), so it works without shelling out to `hdfs dfs -get`.

    IMPORTANT: FileSystem.get(hadoop_conf) alone returns Spark's DEFAULT
    filesystem, which can be file:// even when fs.defaultFS is set to hdfs://
    (depends on how the session was built). We must explicitly request the
    filesystem for the hdfs:// URI, not rely on the default.
    """
    from py4j.java_gateway import java_import

    sc = spark.sparkContext
    java_import(sc._jvm, "org.apache.hadoop.fs.FileSystem")
    java_import(sc._jvm, "org.apache.hadoop.fs.Path")
    java_import(sc._jvm, "java.net.URI")

    hadoop_conf = sc._jsc.hadoopConfiguration()
    src_path = sc._jvm.Path(hdfs_path)
    # Explicitly resolve the filesystem FOR THIS URI, not the JVM's default FS
    fs = sc._jvm.FileSystem.get(sc._jvm.URI(hdfs_path), hadoop_conf)
    fs.copyToLocalFile(False, src_path, sc._jvm.Path(local_path))
    return local_path


def run_inference_partition(rows, local_weights_path: str):
    """
    Runs once per Spark partition. Loads the FINE-TUNED model once (v4
    decision 3 -- same load-once pattern as v3 decision 2), then predicts
    class probabilities for every image/patch in this partition.
    """
    model = load_model(local_weights_path)  # <-- loaded once per partition
    load_marker = id(model)

    results = []
    for row in rows:
        probs = predict_patch(row["content"], model)
        results.append({
            "image_path": row["path"],
            "country": row["country"],
            "label": row["label"],
            "predicted_probs": {k: float(v) for k, v in probs.items()},
            "predicted_class": max(probs, key=probs.get),
            "partition_model_load_marker": float(load_marker % 1_000_000),
        })
    return iter(results)


def check_inference_sanity(result_df):
    """
    Same purpose as v3's embedding sanity check, adapted for classifier
    outputs: catches a degenerate/untrained model or a train/inference
    preprocessing mismatch BEFORE it wastes a full mAP evaluation run.
    """
    dist = result_df.groupBy("predicted_class").count().collect()
    dist_dict = {row["predicted_class"]: row["count"] for row in dist}
    total = sum(dist_dict.values())

    print("\n[sanity] --- inference sanity check ---")
    print(f"[sanity] predicted class distribution: {dist_dict}")

    max_class_frac = max(dist_dict.values()) / total if total else 0
    if max_class_frac > 0.95:
        print(f"[sanity] WARNING: {max_class_frac*100:.1f}% of predictions are the same class. "
              f"Likely a degenerate/untrained model, or train/inference preprocessing mismatch "
              f"(v4 decision 2 -- check the shared preprocessing module is truly identical).")
    else:
        print("[sanity] PASS: predictions show real class variation, not collapsed to one class.")
    print("[sanity] --- end check ---\n")


def run_inference(num_partitions: int = 4, verbose: bool = True):
    spark = (
        SparkSession.builder
        .appName("RDD2022-FineTunedInference")
        .master(f"local[{num_partitions}]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        local_weights = fetch_weights_to_local(spark, MODEL_WEIGHTS_HDFS_PATH)
    except Exception as e:
        spark.stop()
        raise RuntimeError(
            f"Could not fetch trained weights from {MODEL_WEIGHTS_HDFS_PATH}. "
            f"Has Person 1 finished Stage 1 (fine-tuning) and uploaded the .pt "
            f"file to HDFS yet? Original error: {e}"
        )

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
        "path", "content", "country", "label"
    )
    joined_df = joined_df.repartition(num_partitions)

    t0 = time.time()
    result_rdd = joined_df.rdd.mapPartitions(
        lambda rows: run_inference_partition([r.asDict() for r in rows], local_weights)
    )

    schema = StructType([
        StructField("image_path", StringType()),
        StructField("country", StringType()),
        StructField("label", StringType()),
        StructField("predicted_probs", MapType(StringType(), FloatType())),
        StructField("predicted_class", StringType()),
        StructField("partition_model_load_marker", FloatType()),
    ])
    result_df = spark.createDataFrame(result_rdd, schema=schema)
    result_df.write.mode("overwrite").parquet(INFERENCE_OUT_PATH)
    elapsed = time.time() - t0

    n_rows = result_df.count()
    n_distinct_loads = result_df.select("partition_model_load_marker").distinct().count()

    if verbose:
        print(f"[inference] partitions={num_partitions} rows={n_rows} elapsed={elapsed:.2f}s "
              f"throughput={n_rows/elapsed:.1f} img/s")
        print(f"[inference] distinct model-load markers = {n_distinct_loads} "
              f"(should be <= num_partitions={num_partitions} -- proves load-once-per-partition)")
        check_inference_sanity(result_df)

    spark.stop()
    return {"partitions": num_partitions, "rows": n_rows, "elapsed_sec": elapsed}


if __name__ == "__main__":
    # v4 decision 3's pilot test: run small first, before any full run.
    run_inference(num_partitions=4)