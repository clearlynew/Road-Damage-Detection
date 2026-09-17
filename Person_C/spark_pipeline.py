"""
Person 3's deliverable: Spark job that reads raw images + label table from
HDFS, runs embedding extraction via mapPartitions (model loaded once per
partition -- Sec 6 decision #2), and writes embeddings back to HDFS as Parquet.

CURRENTLY WIRED TO STUBS:
  - stub_embedding.StubModel / extract_embedding  stands in for Person 1
  - hdfs_sim/  (local dir) stands in for real HDFS, stands in for Person 2

SWAP POINTS (search for "SWAP" comments) -- these are the only lines that
should need to change once real components land:
  1. import from Person 1's real module instead of stub_embedding
  2. change hdfs_sim/... paths to real hdfs:// URIs
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

# SWAP 1: replace this import with Person 1's real module when ready
from stub_embedding import StubModel, extract_embedding, EMBEDDING_DIM

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
    model = StubModel()  # <-- loaded once per partition, proof: load_count_marker below
    load_marker = model.load_count_marker

    results = []
    for row in rows:
        vec = extract_embedding(row["content"], model)
        results.append({
            "image_path": row["path"],
            "country": row["country"],
            "label": row["label"],
            "embedding": vec.tolist(),
            "partition_model_load_marker": load_marker,  # same value for every row in this partition = proof of "load once"
        })
    return iter(results)


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

    spark.stop()
    return {"partitions": num_partitions, "rows": n_rows, "elapsed_sec": elapsed}


if __name__ == "__main__":
    # This is your Sec 6 decision #2 pilot test: run small first (10-50 images)
    # before any full run. Stub data is 200 rows -- fine for this smoke test.
    run(num_partitions=4)