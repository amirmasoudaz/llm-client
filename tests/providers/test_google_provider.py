"""
Tests for GoogleProvider.
"""

# Mock google.genai if not available
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_client.providers.google import GoogleProvider
from llm_client.providers.types import Message, ToolCall

if "google.genai" not in sys.modules:
    mock_google = MagicMock()
    sys.modules["google"] = mock_google
    sys.modules["google.genai"] = MagicMock()
    sys.modules["google.genai.types"] = MagicMock()


@pytest.fixture
def mock_genai():
    """Mock the genai module."""
    with patch("llm_client.providers.google.genai") as mock:
        # Setup mock client
        mock_client = MagicMock()
        mock.Client.return_value = mock_client

        # Setup async client
        mock_aio = MagicMock()
        mock_client.aio = mock_aio
        mock_aio.models = MagicMock()

        yield mock


@pytest.fixture
def mock_types():
    """Mock the types module."""
    with patch("llm_client.providers.google.types") as mock:
        # Setup Part factory methods
        mock.Part.from_text = MagicMock(side_effect=lambda text: {"text": text})
        mock.Part.from_function_call = MagicMock(
            side_effect=lambda name, args: {"function_call": {"name": name, "args": args}}
        )
        mock.Part.from_function_response = MagicMock(
            side_effect=lambda name, response: {"function_response": {"name": name, "response": response}}
        )
        mock.Content = MagicMock(side_effect=lambda role, parts: {"role": role, "parts": parts})
        mock.FunctionDeclaration = MagicMock(
            side_effect=lambda name, description, parameters_json_schema: {
                "name": name,
                "description": description,
            }
        )
        mock.Tool = MagicMock(side_effect=lambda function_declarations: {"functions": function_declarations})
        mock.GenerateContentConfig = MagicMock()
        mock.AutomaticFunctionCallingConfig = MagicMock()
        yield mock


@pytest.fixture
def provider(mock_genai, mock_types):
    with patch("llm_client.providers.google.GOOGLE_AVAILABLE", True):
        return GoogleProvider(api_key="test-key")


class TestGoogleProvider:
    def test_init_raises_without_api_key(self, mock_genai, mock_types):
        with patch.dict("os.environ", clear=True):
            with pytest.raises(ValueError, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
                with patch("llm_client.providers.google.GOOGLE_AVAILABLE", True):
                    GoogleProvider(api_key=None)

    def test_init_creates_client(self, mock_genai, mock_types):
        with patch("llm_client.providers.google.GOOGLE_AVAILABLE", True):
            GoogleProvider(api_key="test-key")
        mock_genai.Client.assert_called_with(api_key="test-key")

    def test_convert_messages_system_extraction(self, provider, mock_types):
        messages = [
            Message.system("Be helpful"),
            Message.user("Hello"),
        ]
        system_instruction, history = provider._convert_messages(messages)

        assert system_instruction == "Be helpful"
        assert len(history) == 1

    def test_convert_messages_with_tools(self, provider, mock_types):
        messages = [
            Message.assistant("Calling tool", tool_calls=[ToolCall(id="1", name="search", arguments='{"q": "python"}')])
        ]
        _, history = provider._convert_messages(messages)

        assert len(history) == 1
        # Verify Part.from_function_call was called
        mock_types.Part.from_function_call.assert_called()

    @pytest.mark.asyncio
    async def test_complete_basic(self, provider, mock_genai, mock_types):
        # Setup mock response
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Hello there!"
        mock_part.function_call = None

        mock_response.parts = [mock_part]
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]

        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata.total_token_count = 15

        # Setup async mock
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(return_value=mock_response)

        # Execute
        result = await provider.complete(Message.user("Hi"))

        assert result.content == "Hello there!"
        assert result.usage.total_tokens == 15
        assert result.model == provider.model_name

        mock_genai.Client.return_value.aio.models.generate_content.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_complete_with_tool_call(self, provider, mock_genai, mock_types):
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = None

        # Mock function call
        mock_fc = MagicMock()
        mock_fc.name = "get_weather"
        mock_fc.args = {"city": "New York"}
        mock_part.function_call = mock_fc

        mock_response.parts = [mock_part]
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata.total_token_count = 15

        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await provider.complete(Message.user("Weather?"))

        assert result.tool_calls
        assert result.tool_calls[0].name == "get_weather"
        assert '"New York"' in result.tool_calls[0].arguments

    @pytest.mark.asyncio
    async def test_stream(self, provider, mock_genai, mock_types):
        # Create async iterator for stream
        async def mock_stream_gen():
            # Chunk 1: Text
            c1 = MagicMock()
            p1 = MagicMock()
            p1.text = "Hello"
            p1.function_call = None
            c1.candidates = [MagicMock(content=MagicMock(parts=[p1]))]
            c1.usage_metadata = None
            yield c1

            # Chunk 2: Usage
            c2 = MagicMock()
            c2.candidates = []
            c2.usage_metadata = MagicMock()
            c2.usage_metadata.prompt_token_count = 10
            c2.usage_metadata.candidates_token_count = 5
            c2.usage_metadata.total_token_count = 15
            yield c2

        mock_genai.Client.return_value.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream_gen())

        events = []
        async for event in provider.stream(Message.user("Hi")):
            events.append(event)

        assert len(events) == 4  # META, Token, Usage, Done
        assert events[0].type.name == "META"
        assert events[1].data == "Hello"
        assert events[2].type.name == "USAGE"

    @pytest.mark.asyncio
    async def test_embed(self, provider, mock_genai, mock_types):
        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]
        mock_response.embeddings = [mock_embedding]

        mock_genai.Client.return_value.aio.models.embed_content = AsyncMock(return_value=mock_response)

        result = await provider.embed("Hello")

        assert result.embeddings == [[0.1, 0.2, 0.3]]
        mock_genai.Client.return_value.aio.models.embed_content.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close(self, provider, mock_genai):
        # Setup async mock for aclose
        mock_aio = mock_genai.Client.return_value.aio
        mock_aio.aclose = AsyncMock()

        await provider.close()
        mock_aio.aclose.assert_awaited_once()
