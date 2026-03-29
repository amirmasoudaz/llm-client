from llm_client.content import Message
from llm_client.request_builders import build_request_spec, infer_model_name, infer_provider_name


class _FakeModel:
    def __init__(self, key: str) -> None:
        self.key = key


class _FakeProvider:
    def __init__(self) -> None:
        self.model = _FakeModel("fake-model")


def test_request_builder_infers_provider_model_and_extra_fields() -> None:
    provider = _FakeProvider()

    spec = build_request_spec(
        provider=provider,
        messages=[{"role": "user", "content": "hi"}],
        request_kwargs={
            "temperature": 0.2,
            "max_tokens": 64,
            "response_format": "json_object",
            "custom_flag": True,
        },
    )

    assert spec.provider == "fake"
    assert spec.model == "fake-model"
    assert spec.temperature == 0.2
    assert spec.max_tokens == 64
    assert spec.response_format == "json_object"
    assert spec.extra == {"custom_flag": True}
    assert spec.messages == [Message.user("hi")]


def test_request_builder_prefers_explicit_model_override() -> None:
    provider = _FakeProvider()

    spec = build_request_spec(
        provider=provider,
        messages=[Message.user("hi")],
        model="override-model",
        request_kwargs={"model": "ignored-request-model"},
    )

    assert spec.model == "override-model"
    assert infer_provider_name(provider) == "fake"
    assert infer_model_name(provider) == "fake-model"


def test_request_builder_serializes_provider_format_tool_dicts() -> None:
    provider = _FakeProvider()

    spec = build_request_spec(
        provider=provider,
        messages=[Message.user("hi")],
        request_kwargs={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_profile",
                        "description": "Lookup a profile",
                        "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}},
                    },
                }
            ]
        },
    )

    payload = spec.to_dict()
    assert isinstance(payload["tools"], list)
    assert payload["tools"][0]["name"] == "lookup_profile"
    assert payload["tools"][0]["provider_definition"]["type"] == "function"
