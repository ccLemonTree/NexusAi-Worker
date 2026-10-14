"""离线比较同一份 persondog 输出；不调用模型、不写 Kafka。"""

import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.infer.Triton_model.yolov5.utils.processing import postprocess


def benchmark(data, index):
    output = data[f"output_{index}"]
    width, height = int(data[f"width_{index}"]), int(data[f"height_{index}"])
    input_height, input_width = int(data["input_height"]), int(data["input_width"])
    labels = [f"class_{i}" for i in range(output.shape[-1] - 5)]
    rows = []
    for threshold in [0.2, 0.5, 0.8]:
        results, timings = {}, {}
        for name, prefilter in [("original", 0.0), ("optimized", max(0.0, threshold - 0.005))]:
            elapsed = []
            for _ in range(3):
                start = time.perf_counter()
                boxes = postprocess(output, width, height, [input_height, input_width], prefilter, 0.2, labels)
                selected = [box.dict() for box in boxes if box.confidence > threshold]
                elapsed.append((time.perf_counter() - start) * 1000)
            results[name] = selected
            timings[name] = round(statistics.median(elapsed), 3)
        assert results["original"] == results["optimized"], "optimized detections differ"
        rows.append({
            "threshold": threshold, "same_results": True,
            "raw_candidates": int(output.shape[1]),
            "prefilter_candidates": int(np.count_nonzero(output[0, :, 4] >= max(0.0, threshold - 0.005))),
            "retained": len(results["optimized"]), "median_ms": timings,
        })
    return {"sample": index, "results": rows}


if __name__ == "__main__":
    for path in sys.argv[1:]:
        with np.load(path) as data:
            for index in sorted(key[7:] for key in data.files if key.startswith("output_")):
                print(json.dumps(benchmark(data, index), ensure_ascii=False))
