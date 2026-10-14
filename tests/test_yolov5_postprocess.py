import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from api.infer.Triton_model.yolov5.utils import processing


def load_detector():
    package = ModuleType("tritonclient")
    package.__path__ = []
    grpc = ModuleType("tritonclient.grpc")
    grpc.InferInput = lambda *args: SimpleNamespace(set_data_from_numpy=lambda data: None)
    grpc.InferRequestedOutput = lambda *args: None
    path = Path(__file__).parents[1] / "api/infer/Triton_model/yolov5/yolov5_detector.py"
    spec = importlib.util.spec_from_file_location("_test_yolov5_detector", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"tritonclient": package, "tritonclient.grpc": grpc}):
        spec.loader.exec_module(module)
    return module


class Yolov5PostprocessTests(unittest.TestCase):
    def setUp(self):
        self.detector = load_detector()
        self.image = np.zeros((640, 640, 3), dtype=np.uint8)
        self.config = {"yolov5_persondog": {
            "input": [{"name": "images", "dims": [1, 3, 640, 640], "data_type": "TYPE_FP32"}],
            "output": [{"name": "output"}], "model_version": "1",
            "label_names": ["person", "dog"], "conf_thres": 0.5, "iou_thres": 0.2,
        }}

    def infer(self, prediction, rules):
        self.reads = []
        def as_numpy(name):
            self.reads.append(name)
            return prediction
        client = SimpleNamespace(
            is_model_ready=lambda *args: True,
            infer=lambda **kwargs: SimpleNamespace(as_numpy=as_numpy),
        )
        return self.detector.yolov5(client, "yolov5_persondog", self.config, self.image, rules, None)[0]

    def baseline(self, prediction, rules):
        boxes = processing.postprocess(prediction, 640, 640, [640, 640], 0.0, 0.2, ["person", "dog"])
        return [b for b in boxes if b.classname in rules and b.confidence > rules[b.classname]["conf"]]

    def assert_same_boxes(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for a, b in zip(expected, actual):
            self.assertEqual(a.classID, b.classID)
            self.assertEqual(a.confidence, b.confidence)
            np.testing.assert_array_equal([a.u1, a.v1, a.u2, a.v2], [b.u1, b.v1, b.u2, b.v2])

    def test_prefilter_uses_lowest_requested_threshold_not_zero_or_model_default(self):
        rules = {"person": {"conf": 0.8}, "dog": {"conf": 0.2}}
        prediction = np.array([[[50, 50, 20, 20, 0.3, 0.0, 1.0]]], dtype=np.float32)
        with patch.object(self.detector, "postprocess", wraps=self.detector.postprocess) as post:
            boxes = self.infer(prediction, rules)
        self.assertGreater(post.call_args.args[4], 0.19)
        self.assertLessEqual(post.call_args.args[4], 0.2)
        self.assertEqual(len(boxes), 1)

    def test_output_is_deserialized_only_once(self):
        self.infer(np.zeros((1, 2, 7), dtype=np.float32), {"person": {"conf": 0.8}})
        self.assertEqual(self.reads, ["output"])

    def test_preserves_existing_cross_class_suppression(self):
        prediction = np.array([[
            [100, 100, 30, 30, 0.95, 0, 1],
            [100, 100, 30, 30, 0.9, 1, 0],
        ]], dtype=np.float32)
        rules = {"person": {"conf": 0.8}}
        self.assertEqual(self.baseline(prediction, rules), [])
        self.assertEqual(self.infer(prediction, rules), [])

    def test_prefilter_preserves_confidence_rounding_before_final_threshold(self):
        prediction = np.array([[[100, 100, 30, 30, 0.5951, 1, 0]]], dtype=np.float32)
        rules = {"person": {"conf": 0.599}}
        expected = self.baseline(prediction.copy(), rules)
        self.assertEqual(len(expected), 1)
        self.assert_same_boxes(expected, self.infer(prediction.copy(), rules))

    def test_threshold_boundaries_zero_threshold_and_individual_rules_match_baseline(self):
        prediction = np.array([[
            [20, 20, 10, 10, 0, 1, 0], [50, 50, 10, 10, 0.2, 0, 1],
            [80, 80, 10, 10, 0.5, 1, 0], [110, 110, 10, 10, 0.8, 1, 0],
            [140, 140, 10, 10, 0.9, 1, 0],
        ]], dtype=np.float32)
        for rules in [{"person": {"conf": 0.8}, "dog": {"conf": 0.2}},
                      {"person": {"conf": 0.0}}, {"person": {"conf": -0.1}}]:
            with self.subTest(rules=rules):
                self.assert_same_boxes(self.baseline(prediction.copy(), rules), self.infer(prediction.copy(), rules))

    def test_random_predictions_match_original_results_including_ties(self):
        rng = np.random.default_rng(402)
        for _ in range(8):
            prediction = rng.random((1, 400, 7), dtype=np.float32)
            prediction[:, :, :2] *= 640
            prediction[:, :, 2:4] *= 200
            prediction[:, :, 4] = np.round(prediction[:, :, 4], 1)
            rules = {"person": {"conf": 0.7}, "dog": {"conf": 0.2}}
            self.assert_same_boxes(self.baseline(prediction.copy(), rules), self.infer(prediction.copy(), rules))


if __name__ == "__main__":
    unittest.main()
