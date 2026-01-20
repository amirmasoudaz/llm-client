import unittest

from llm_client.serialization import stable_json_dumps


class SerializationTests(unittest.TestCase):
    def test_stable_json_dumps_is_deterministic(self) -> None:
        a = {"b": 1, "a": {"z": 3, "y": 2}}
        b = {"a": {"y": 2, "z": 3}, "b": 1}
        self.assertEqual(stable_json_dumps(a), stable_json_dumps(b))


if __name__ == "__main__":
    unittest.main()
