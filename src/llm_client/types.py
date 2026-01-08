from typing import TypedDict, Optional, Any, Union, Literal, List
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

class RequestParams(TypedDict, total=False):
    model: str
    messages: List[ChatCompletionMessageParam]
    temperature: Optional[float]
    top_p: Optional[float]
    n: Optional[int]
    stream: Optional[bool]
    stop: Optional[Union[str, List[str]]]
    max_tokens: Optional[int]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    logit_bias: Optional[dict]
    user: Optional[str]
    response_format: Optional[Any]
    seed: Optional[int]
    tools: Optional[List[ChatCompletionToolParam]]
    tool_choice: Optional[Union[str, dict]]
    reasoning_effort: Optional[str]
    # Internal usage
    stream_mode: Optional[str] # "pusher" or "sse"
    stream_options: Optional[dict]
