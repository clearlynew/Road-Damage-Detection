"""
Person 3's deliverable: scaling experiment (Sec 6 decision #5).
Runs the pipeline at 1, 2, 4, 8 partitions/cores, logs wall-clock time,
throughput, and parallel efficiency, writes scaling_results.csv.

v4 UPDATE: this currently targets the ABLATION track (spark_pipeline_ablation.py,
frozen embeddings), because that runs standalone with no trained model needed.
Once Person 1's real .pt lands and is in HDFS, switch the import below to
`from spark_pipeline import run_inference as run` to scale-test the PRIMARY
(fine-tuned) track instead -- same harness, just a different function.

NOTE: on this small stub dataset (200 rows), don't expect real speedup --
Spark overhead dominates at this scale. Meaningful only on the real
~37,000-image run (Sec 6 decision #5/#6).
"""
import csv
from spark_pipeline_ablation import run

PARTITION_LEVELS = [1, 2, 4, 8]
OUT_CSV = "scaling_results.csv"


def main():
    results = []
    baseline_time = None

    for p in PARTITION_LEVELS:
        print(f"\n=== Running with {p} partition(s) ===")
        r = run(num_partitions=p, verbose=True)
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
              f"throughput={r['throughput_img_per_sec']:.1f} img/s "
              f"speedup={r['speedup']:.2f}x efficiency={r['parallel_efficiency']:.2f}")


if __name__ == "__main__":
    main()