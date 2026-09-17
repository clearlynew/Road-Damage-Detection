# Person 3 — Spark Pipeline & Scaling (RDD2022)

Status: fully built and tested end-to-end against real HDFS (localhost:9000)
on a Linux + Hadoop 3.4.1 + Spark 3.5.5 + Java 8 setup. Currently running
against STUB data (fake images, fake embeddings) standing in for Person 1
(embeddings) and Person 2 (real dataset). Swap points are marked in the code
and described below.

## Environment (tested working combination)

- Java 8 (Spark 3.5.5 does NOT support Java 24; use `update-alternatives --config java`
  if multiple JDKs are installed)
- pyspark 3.5.5 (`pip install "pyspark==3.5.5"` — do NOT use plain `pip install pyspark`,
  which grabs Spark 4.x and requires Java 17+)
- A Python venv is recommended:
    python3 -m venv ~/rdd_venv
    source ~/rdd_venv/bin/activate
    pip install "pyspark==3.5.5" pandas pyarrow pillow numpy
- Real HDFS running (`start-dfs.sh`, confirm with `jps` -> NameNode + DataNode,
  and `hdfs dfs -ls /`)
- Find your HDFS address with: `hdfs getconf -confKey fs.defaultFS`
  (scripts below assume `hdfs://localhost:9000` — edit the two path constants
  in spark_pipeline.py if yours differs)

## Run order

    # 1. Generate fake data locally (stands in for Person 2)
    python3 generate_stub_data.py

    # 2. Push the fake raw images + label table into real HDFS
    hdfs dfs -mkdir -p /rdd2022/raw_images
    hdfs dfs -mkdir -p /rdd2022/label_table
    hdfs dfs -put -f hdfs_sim/raw_images/* /rdd2022/raw_images/
    hdfs dfs -put -f hdfs_sim/label_table/labels.parquet /rdd2022/label_table/

    # 3. Run the pilot test / smoke test (Sec 6 decision #2 -- the "biggest
    #    known pitfall" check). Reads + writes real HDFS.
    python3 spark_pipeline.py

    # 4. Confirm output landed correctly
    hdfs dfs -ls /rdd2022/embeddings

    # 5. Run the full scaling experiment (Sec 6 decision #5): 1/2/4/8 partitions,
    #    logs wall-clock, throughput, parallel efficiency -> scaling_results.csv
    python3 scaling_experiment.py

All five steps have been run successfully on real HDFS as of this README's
last update. Confirmed output: 200 rows processed, 4 correctly-partitioned
Parquet files + _SUCCESS marker in /rdd2022/embeddings, and a full
scaling_results.csv across all four partition levels.

## Files

- `stub_embedding.py` — fake version of Person 1's `extract_embedding(image) -> vector`.
  Same output shape (1280-dim) as real MobileNetV2, so downstream schemas won't change
  when the real one lands.
- `generate_stub_data.py` — fake version of Person 2's raw images + label table.
  Writes to local `hdfs_sim/` first; you then `hdfs dfs -put` it into real HDFS
  (see step 2 above). Real data from Person 2 will replace this step entirely —
  you'll `hdfs dfs -put` her real JPEGs + label table instead.
- `spark_pipeline.py` — YOUR real deliverable. Reads raw images directly from
  HDFS via Spark's `binaryFile` source (NOT plain Python `open()`, which can't
  read `hdfs://` paths), joins them with the label table by filename, runs
  `mapPartitions` with the model loaded once per partition, writes embeddings
  back to HDFS as Parquet. Two clearly marked `SWAP` points for Person 1's
  real module and (if needed) a different HDFS address.
- `scaling_experiment.py` — YOUR real deliverable. Calls `spark_pipeline.py`'s
  `run()` at 1/2/4/8 partitions, logs wall-clock/throughput/parallel efficiency,
  writes `scaling_results.csv`.

## Swap-in points when real deliverables land

1. **Person 1's real model:** in `spark_pipeline.py`, replace
   `from stub_embedding import StubModel, extract_embedding, EMBEDDING_DIM`
   with her real module. In `process_partition()`, replace `StubModel()` and
   the `extract_embedding()` call with her real load-once and inference calls —
   keep the same load-once-per-partition *shape*. Check what input format her
   function expects (raw bytes vs. PIL Image vs. tensor) — you may need a small
   conversion step, since your rows currently carry raw bytes (`row["content"]`)
   read via Spark's binaryFile source.
2. **Person 2's real data:** skip `generate_stub_data.py` entirely. Instead,
   `hdfs dfs -put` her real JPEGs into `/rdd2022/raw_images/<country>/` and her
   real label table Parquet into `/rdd2022/label_table/`. No code changes needed
   in `spark_pipeline.py` as long as the label table has an `image_path` (or
   any column ending in the image filename) and a `label` column, `country`
   column — matching the stub schema.
3. **After swapping either one**, re-run the pilot test (`python3 spark_pipeline.py`)
   first — this is your real Sec 6 decision #2 check, not the inconclusive one
   from the local-only stub test earlier. Then re-run `scaling_experiment.py`
   for the real, reportable scaling curve (needed for Days 8-9 / the report).

## Known non-issues (ignore these if you see them again)

- `WARN FileStreamSink: Assume no metadata directory...` / a
  `FileNotFoundException` that flashes by mid-run — this is Spark internally
  probing whether the path is a streaming source. Harmless, job still
  completes correctly.
- `WARN MemoryManager: Total allocation exceeds 95%...` at higher partition
  counts (e.g. 8) — Parquet writers sharing memory tightly. Not an error.
- `WARN NativeCodeLoader: Unable to load native-hadoop library...` — cosmetic,
  Spark falls back to Java implementations, no functional impact.

## Known real finding (do NOT ignore this one)

The "distinct model-load markers" check in `spark_pipeline.py`'s output
(intended to prove the model loads once per partition, per decision #2) has
consistently returned 1 regardless of partition count, both locally and
against real HDFS. This is either a Spark local-mode scheduling artifact, or
a real bug in how the marker is captured. It has NOT yet been validated
correctly. Re-check this explicitly once Person 1's real (non-trivial-cost)
model is swapped in — a real model's load time should make partition-level
separation actually visible if the pattern is working. If it still shows 1
distinct marker with a real model, investigate further before trusting the
"model loaded once per partition" claim in the report.

## Stub scaling numbers are NOT the real result

The scaling_results.csv produced against 200 stub images showed efficiency
dropping sharply at 4 and 8 partitions (0.54, 0.28) — this is expected at this
tiny scale (Spark's own overhead dominates) and matches what Sec 6 decision #5
already predicts: real speedup only becomes visible at the full ~37,000-image
volume. Do not use these numbers in the report or demo. Re-run once real data
is in place.