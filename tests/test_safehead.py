import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace


class FakeModelClass:
    pass


class FakeBoundingBox:
    pass


def load_safehead_module():
    fake_class_info = ModuleType("api.infer.Utils.class_info")
    fake_class_info.ModelClass = FakeModelClass

    fake_boundingbox = ModuleType("api.infer.Utils.boundingbox")
    fake_boundingbox.BoundingBox = FakeBoundingBox

    fake_init = ModuleType("tools.init")
    fake_init.cfg = SimpleNamespace()

    fake_logger_tools = ModuleType("tools.logger_tools")
    fake_logger_tools.CangQiong_Smart_Model_logger = SimpleNamespace(
        error=lambda *args, **kwargs: None
    )

    fake_running = ModuleType("api.infer.running")
    fake_running.analyseRun = lambda *args, **kwargs: []

    replacements = {
        "api.infer.Utils.class_info": fake_class_info,
        "api.infer.Utils.boundingbox": fake_boundingbox,
        "tools.init": fake_init,
        "tools.logger_tools": fake_logger_tools,
        "api.infer.running": fake_running,
    }
    previous_modules = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)

    try:
        safehead_path = (
            Path(__file__).parents[1]
            / "api/infer/Model_pipline/Safehead/Safehead.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_safehead_under_test",
            safehead_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


safehead_module = load_safehead_module()
filter_unmatched_first_boxes = safehead_module.filter_unmatched_first_boxes


def make_box(x1, y1, x2, y2):
    return SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2)


def test_filters_first_box_that_intersects_second_box():
    first = make_box(10, 10, 30, 30)
    second = make_box(20, 20, 40, 40)

    assert filter_unmatched_first_boxes([first], [second]) == []


def test_filters_first_box_when_second_box_is_within_twenty_percent_margin():
    first = make_box(10, 10, 30, 30)
    second = make_box(33, 12, 40, 20)

    assert filter_unmatched_first_boxes([first], [second]) == []


def test_keeps_first_box_when_second_box_is_outside_twenty_percent_margin():
    first = make_box(10, 10, 30, 30)
    second = make_box(35, 12, 40, 20)

    assert filter_unmatched_first_boxes([first], [second]) == [first]


def test_filters_first_box_when_second_box_touches_expanded_edge():
    first = make_box(10, 10, 30, 30)
    second = make_box(34, 12, 40, 20)

    assert filter_unmatched_first_boxes([first], [second]) == []


def test_keeps_all_first_boxes_when_second_detection_is_empty():
    first_boxes = [make_box(10, 10, 30, 30), make_box(50, 50, 70, 70)]

    assert filter_unmatched_first_boxes(first_boxes, []) == first_boxes


def test_execute_runs_second_label_once_on_whole_image_and_returns_unmatched_first_boxes(monkeypatch):
    picture = object()
    near_first = make_box(10, 10, 30, 30)
    far_first = make_box(100, 100, 120, 120)
    second = make_box(20, 20, 40, 40)
    calls = []

    safehead_module.cfg.logicModelDict = {
        "Safehead-person": {1: {"label": ["yolov5nohead"]}}
    }

    def fake_analyse_run(labels, images, camera_info):
        calls.append((labels, images, camera_info))
        return [second]

    monkeypatch.setattr(safehead_module, "analyseRun", fake_analyse_run)

    model = safehead_module.Model()
    model.logicModelName = "Safehead-person"
    model.logicResult = [near_first, far_first]
    model.picture = picture
    model.cameraInfo = object()

    result, images = model.execute()

    assert result == [far_first]
    assert images == []
    assert far_first.classname == "Safehead-person"
    assert calls == [(["yolov5nohead"], [picture, picture], model.cameraInfo)]
