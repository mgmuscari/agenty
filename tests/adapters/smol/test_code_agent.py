from unittest.mock import Mock

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_core import ValidationError

from agenty.integrations.smol.code_agent import SmolCodeAgent

from smolagents.default_tools import PythonInterpreterTool

from .fake_models import fake_code_model_no_return, fake_code_model


@pytest.mark.asyncio
async def test_code_str_str():
    agent: SmolCodeAgent[str, str] = SmolCodeAgent(
        model=Mock(spec=TestModel),
        smol_tools=[PythonInterpreterTool()],
        smol_verbosity_level=0,
        input_schema=str,
        output_schema=str,
    )
    agent.get_smol_model = Mock(return_value=fake_code_model)
    resp = await agent.run("What is 2 multiplied by 3.6452?")

    assert isinstance(resp, str)
    assert resp == "7.2904"


@pytest.mark.asyncio
async def test_code_str_float():
    agent: SmolCodeAgent[str, float] = SmolCodeAgent(
        model=Mock(spec=TestModel),
        smol_tools=[PythonInterpreterTool()],
        smol_verbosity_level=0,
        input_schema=str,
        output_schema=float,
    )
    agent.get_smol_model = Mock(return_value=fake_code_model)
    resp = await agent.run("What is 2 multiplied by 3.6452?")

    assert isinstance(resp, float)
    assert resp == 7.2904


@pytest.mark.asyncio
async def test_code_str_int():
    agent: SmolCodeAgent[str, int] = SmolCodeAgent(
        model=Mock(spec=TestModel),
        smol_tools=[PythonInterpreterTool()],
        smol_verbosity_level=0,
        input_schema=str,
        output_schema=int,
    )
    agent.get_smol_model = Mock(return_value=fake_code_model)
    resp = await agent.run("What is 2 multiplied by 3.6452?")

    assert isinstance(resp, int)
    assert resp == 7


@pytest.mark.asyncio
async def test_code_no_return():
    agent: SmolCodeAgent[str, float] = SmolCodeAgent(
        model=Mock(spec=TestModel),
        smol_tools=[PythonInterpreterTool()],
        smol_verbosity_level=0,
        input_schema=str,
        output_schema=float,
    )
    agent.get_smol_model = Mock(return_value=fake_code_model_no_return)

    with pytest.raises(ValidationError):
        pass
        await agent.run("What is 2 multiplied by 3.6452?")
