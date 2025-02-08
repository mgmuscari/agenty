import pytest
from typing import Optional
from pydantic import TypeAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ModelResponsePart,
    ToolCallPart,
)
from agenty.models import allow_model_requests, AgentInfo, FunctionModel
from agenty.types import BaseIO

allow_model_requests(False)


class FakeInput(BaseIO):
    query: str
    context: Optional[str] = None


class FakeOutput(BaseIO):
    response: str
    confidence: float


@pytest.fixture
def str_test_model() -> FunctionModel:
    async def test_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        response_parts: list[ModelResponsePart] = []

        def process_request(message: ModelRequest):
            for part in message.parts:
                match part.part_kind:
                    case "user-prompt":
                        if "agenty" in part.content:
                            response_parts.append(TextPart(f"success: {part.content}"))
                        else:
                            response_parts.append(TextPart(f"failure: {part.content}"))
                    case _:
                        pass

        for message in messages:
            if message.kind == "request":
                process_request(message)

        return ModelResponse(
            parts=response_parts,
            model_name="agenty-test-str",
        )

    return FunctionModel(test_func)


@pytest.fixture
def schema_test_model() -> FunctionModel:
    async def test_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        response_parts: list[ModelResponsePart] = []

        def process_request(message: ModelRequest):
            for part in message.parts:
                match part.part_kind:
                    case "user-prompt":
                        response_str = "failure"
                        if "agenty" in part.content:
                            response_str = "success"
                        input = TypeAdapter(FakeInput).validate_json(part.content)
                        output = FakeOutput(
                            response=f"{response_str}: {input.query}",
                            confidence=float(input.context) if input.context else -1.0,
                        )
                        response_parts.append(
                            ToolCallPart(
                                tool_name="final_result",
                                args=output.model_dump(),
                                tool_call_id=None,
                            )
                        )
                    case _:
                        pass

        for message in messages:
            if message.kind == "request":
                process_request(message)

        return ModelResponse(
            parts=response_parts,
            model_name="agenty-test-schema",
        )

    return FunctionModel(test_func)


@pytest.fixture
def error_test_model() -> FunctionModel:
    async def test_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        response_parts: list[ModelResponsePart] = []

        def process_request(message: ModelRequest):
            for part in message.parts:
                match part.part_kind:
                    case "user-prompt":
                        TypeAdapter(FakeInput).validate_json(part.content)
                        response_parts.append(
                            ToolCallPart(
                                tool_name="final_result",
                                args={"error": "bad output"},
                                tool_call_id=None,
                            )
                        )
                    case _:
                        pass

        for message in messages:
            if message.kind == "request":
                process_request(message)

        return ModelResponse(
            parts=response_parts,
            model_name="agenty-test-error",
        )

    return FunctionModel(test_func)
