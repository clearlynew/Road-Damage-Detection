"""
Person 3's deliverable: scaling experiment (Sec 6 decision #5).
Runs the pipeline at 1, 2, 4, 8 partitions/cores, logs wall-clock time,
throughput, and parallel efficiency, writes scaling_results.csv.

FIX vs. the previous version: the old script reused spark_pipeline_ablation's
run(), which INNER-JOINS every image against the label table. That's wrong
for scaling specifically -- the full local dataset (~38,000 images) is much
bigger than the label table's ~10,000-image ground-truth coverage, so an
inner join would silently drop ~28,000 images and understate throughput.
Scaling doesn't need ground truth at all (it's measuring raw pipeline
speed, not accuracy) -- this version reads every image in
RAW_IMAGES_FULL_PATH directly, no join, no label table.

Choose which track to scale-test with TRACK below:
  "primary" -- fine-tuned model (spark_pipeline.py's architecture),
               requires MODEL_WEIGHTS_HDFS_PATH to already exist in HDFS.
  "ablation" -- frozen MobileNetV2 + embeddings (no trained weights needed,
                usable even before Stage 1 fine-tuning is done).
"""
import io
import os
import csv
import time

os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

if os.name == "nt" and "HADOOP_HOME" not in os.environ:
    os.environ["HADOOP_HOME"] = r"C:\hadoop"
    os.environ["PATH"] = os.environ["HADOOP_HOME"] + r"\bin;" + os.environ["PATH"]

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType, ArrayType, MapType
)
from PIL import Image

TRACK = "ablation"  # "primary" or "ablation" -- see module docstring

RAW_IMAGES_FULL_PATH = "hdfs://localhost:9000/rdd2022/raw_images_full/*/*.jpg"
MODEL_WEIGHTS_HDFS_PATH = "hdfs://localhost:9000/models/rdamage_mobilenetv2.pt"
PARTITION_LEVELS = [1, 2, 4, 8]
OUT_CSV = "scaling_results.csv"

RESIZE_LONG_SIDE = 1024
PATCH_SIZE = 224
STRIDE = 112


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


def _scaling_partition_ablation(rows):
    from extract_embedding import load_model, extract_embedding
    model = load_model()
    load_marker = id(model)
    results = []
    for row in rows:
        try:
            img = Image.open(io.BytesIO(row["content"])).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        scale = get_resize_scale(w, h)
        if scale != 1.0:
            img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
        rw, rh = img.size
        for (x0, y0, x1, y1) in sliding_window_coords(rw, rh):
            crop = img.crop((x0, y0, x1, y1))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG")
            vec = extract_embedding(buf.getvalue(), model)
            results.append({
                "image_path": row["path"], "embedding": vec.tolist(),
                "partition_model_load_marker": float(load_marker % 1_000_000),
            })
    return iter(results)


def _scaling_partition_primary(rows, local_weights_path):
    from model_api import load_model, predict_patch
    model = load_model(local_weights_path)
    load_marker = id(model)
    results = []
    for row in rows:
        try:
            img = Image.open(io.BytesIO(row["content"])).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        scale = get_resize_scale(w, h)
        if scale != 1.0:
            img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
        rw, rh = img.size
        for (x0, y0, x1, y1) in sliding_window_coords(rw, rh):
            crop = img.crop((x0, y0, x1, y1))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG")
            probs = predict_patch(buf.getvalue(), model)
            results.append({
                "image_path": row["path"],
                "predicted_class": max(probs, key=probs.get),
                "partition_model_load_marker": float(load_marker % 1_000_000),
            })
    return iter(results)


def _fetch_weights_to_local(spark, hdfs_path, local_path="/tmp/model_weights.pt"):
    from py4j.java_gateway import java_import
    sc = spark.sparkContext
    java_import(sc._jvm, "org.apache.hadoop.fs.FileSystem")
    java_import(sc._jvm, "org.apache.hadoop.fs.Path")
    java_import(sc._jvm, "java.net.URI")
    hadoop_conf = sc._jsc.hadoopConfiguration()
    src_path = sc._jvm.Path(hdfs_path)
    fs = sc._jvm.FileSystem.get(sc._jvm.URI(hdfs_path), hadoop_conf)
    fs.copyToLocalFile(False, src_path, sc._jvm.Path(local_path))
    return local_path


def run_scaling(num_partitions: int, track: str = TRACK, verbose: bool = True):
    """
    Full-dataset, NO ground-truth join. Reads every image under
    RAW_IMAGES_FULL_PATH directly -- this is the correct scaling scope
    (Sec 6 decision #5/#6: meaningful only at the full ~37,000+-image volume).
    """
    spark = (
        SparkSession.builder
        .appName(f"RDD2022-Scaling-{track}")
        .master(f"local[{num_partitions}]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    images_df = (
        spark.read.format("binaryFile")
        .load(RAW_IMAGES_FULL_PATH)
        .selectExpr("path", "content")
        .repartition(num_partitions)
    )

    if track == "primary":
        local_weights = _fetch_weights_to_local(spark, MODEL_WEIGHTS_HDFS_PATH)
        schema = StructType([
            StructField("image_path", StringType()),
            StructField("predicted_class", StringType()),
            StructField("partition_model_load_marker", FloatType()),
        ])
        t0 = time.time()
        result_rdd = images_df.rdd.mapPartitions(
            lambda rows: _scaling_partition_primary([r.asDict() for r in rows], local_weights)
        )
    elif track == "ablation":
        schema = StructType([
            StructField("image_path", StringType()),
            StructField("embedding", ArrayType(FloatType())),
            StructField("partition_model_load_marker", FloatType()),
        ])
        t0 = time.time()
        result_rdd = images_df.rdd.mapPartitions(
            lambda rows: _scaling_partition_ablation([r.asDict() for r in rows])
        )
    else:
        spark.stop()
        raise ValueError(f"Unknown track: {track!r}. Use 'primary' or 'ablation'.")

    result_df = spark.createDataFrame(result_rdd, schema=schema)
    n_rows = result_df.count()  # forces the actual computation for timing
    elapsed = time.time() - t0

    if verbose:
        print(f"[scaling:{track}] partitions={num_partitions} patches={n_rows} "
              f"elapsed={elapsed:.2f}s throughput={n_rows/elapsed:.1f} patches/s")

    spark.stop()
    return {"partitions": num_partitions, "rows": n_rows, "elapsed_sec": elapsed}


def main():
    print(f"Scaling track: {TRACK}")
    print(f"Reading from: {RAW_IMAGES_FULL_PATH}\n")

    results = []
    baseline_time = None

    for p in PARTITION_LEVELS:
        print(f"\n=== Running with {p} partition(s) ===")
        r = run_scaling(num_partitions=p, track=TRACK, verbose=True)
        if baseline_time is None:
            baseline_time = r["elapsed_sec"]  # 1-partition run is the baseline
        speedup = baseline_time / r["elapsed_sec"]
        efficiency = speedup / p
        r["speedup"] = speedup
        r["parallel_efficiency"] = efficiency
        r["throughput_img_per_sec"] = r["rows"] / r["elapsed_sec"]
        results.append(r)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "partitions", "rows", "elapsed_sec", "throughput_img_per_sec",
            "speedup", "parallel_efficiency"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n[scaling] wrote results to {OUT_CSV}")
    for r in results:
        print(f"  partitions={r['partitions']:<3} elapsed={r['elapsed_sec']:.2f}s "
              f"throughput={r['throughput_img_per_sec']:.1f} patches/s "
              f"speedup={r['speedup']:.2f}x efficiency={r['parallel_efficiency']:.2f}")


if __name__ == "__main__":
    main()