import unittest

try:
    from jsonschema import validate as _jsonschema_validate
except Exception:
    _jsonschema_validate = None

from llm_client.tools.base import Tool


@unittest.skipIf(_jsonschema_validate is None, "jsonschema not installed")
class ToolValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_validation_rejects_invalid_args(self) -> None:
        async def handler(x: int) -> int:
            return x

        tool = Tool(
            name="test",
            description="test tool",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            handler=handler,
            strict=True,
        )

        result = await tool.execute_json('{"x": "nope"}')
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    async def test_strict_validation_allows_valid_args(self) -> None:
        async def handler(x: int) -> int:
            return x + 1

        tool = Tool(
            name="test",
            description="test tool",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            handler=handler,
            strict=True,
        )

        result = await tool.execute_json('{"x": 1}')
        self.assertTrue(result.success)
        self.assertEqual(result.content, "2")


if __name__ == "__main__":
    unittest.main()
