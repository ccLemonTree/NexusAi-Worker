import unittest
from pathlib import Path


class WorkerProjectBoundaryTests(unittest.TestCase):
    def test_worker_application_has_no_kafka_client_code(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "worker_main.py").is_file())

        forbidden = ("aiokafka", "AIOKafka", "KafkaConsumer", "kafka_handler")
        sources = (
            root / "worker_main.py",
            *(path for directory in ("inference", "api", "tools", "utils")
              for path in (root / directory).rglob("*.py")),
        )
        content = "\n".join(path.read_text(encoding="utf-8") for path in sources)
        self.assertFalse(any(token in content for token in forbidden))


if __name__ == "__main__":
    unittest.main()
