# tests/test_prompt_runner.py
import sys
sys.path.append('.')
sys.path.append('langdspy')
import os
import dotenv
dotenv.load_dotenv()
import pytest
from unittest.mock import MagicMock
from langchain.chains import LLMChain
from langdspy import PromptRunner, DefaultPromptStrategy, InputField, OutputField, Model, PromptSignature, Prediction

class TestPromptSignature(PromptSignature):
    input = InputField(name="input", desc="Input field")
    output = OutputField(name="output", desc="Output field")

class TestModel(Model):
    test_prompt_runner = PromptRunner(template_class=TestPromptSignature, prompt_strategy=DefaultPromptStrategy)

    def invoke(self, input_dict, config):
        print(f"Invoked with input {input_dict} and config {config}")
        result = self.test_prompt_runner.invoke(input_dict, config=config)
        print(result)
        return result.output

from unittest.mock import patch

# def test_print_prompt_in_config():
#     model = TestModel(n_jobs=1, print_prompt=True)

#     input_dict = {"input": "Test input"}
#     mock_invoke = MagicMock(return_value=Prediction(**{**input_dict, "output": "Test output"}))

#     # with patch.object(TestModel, 'invoke', new=mock_invoke):
#     config = {"llm": mock_invoke}
#     result = model.invoke(input_dict, config)
#     print(result)

#     mock_invoke.assert_called_once_with(input_dict, config)
#     assert "print_prompt" in config
#     assert config["print_prompt"] == True
#     assert result.output == "Test output"

from langchain.chat_models.base import BaseChatModel

class FakeLLM(BaseChatModel):
    def invoke(self, *args, **kwargs):
        return "INVOKED"

    def _generate(self, *args, **kwargs):
        return None

    def _llm_type(self) -> str:
        return "fake"

def test_print_prompt_in_inputs():
    model = TestModel(n_jobs=1, print_prompt="TEST")
    input_dict = {"input": "Test input"}
    mock_invoke = MagicMock(return_value="FORMATTED PROMPT")
    
    with patch.object(DefaultPromptStrategy, 'format_prompt', new=mock_invoke):
        config = {"llm": FakeLLM()}
        result = model.invoke(input_dict, config=config)

        print(result)
        print(f"Called with {mock_invoke.call_count} {mock_invoke.call_args_list} {mock_invoke.call_args}")
        call_args = {**input_dict, 'print_prompt': "TEST", 'trained_state': model.trained_state, 'use_training': True}
        print(f"Expecting call {call_args}")
        mock_invoke.assert_called_once_with(**call_args)

def test_trained_state_in_inputs():
    model = TestModel(n_jobs=1)
    input_dict = {"input": "Test input"}
    mock_invoke = MagicMock(return_value="FORMATTED PROMPT")
    
    with patch.object(DefaultPromptStrategy, 'format_prompt', new=mock_invoke):
        config = {"llm": FakeLLM()}
        model.trained_state.examples = [("EXAMPLE_X", "EXAMPLE_Y")]
        result = model.invoke(input_dict, config=config)

        print(result)
        print(f"Called with {mock_invoke.call_count} {mock_invoke.call_args_list} {mock_invoke.call_args}")
        call_args = {**input_dict, 'print_prompt': "TEST", 'trained_state': model.trained_state, 'use_training': True}
        print(f"Expecting call {call_args}")
        mock_invoke.assert_called_once_with(**call_args)

def test_use_training():
    model = TestModel(n_jobs=1)
    input_dict = {"input": "Test input"}
    mock_invoke = MagicMock(return_value="FORMATTED PROMPT")
    
    with patch.object(DefaultPromptStrategy, 'format_prompt', new=mock_invoke):
        config = {"llm": FakeLLM(), "use_training": False}
        model.trained_state.examples = [("EXAMPLE_X", "EXAMPLE_Y")]
        result = model.invoke(input_dict, config=config)

        print(result)
        print(f"Called with {mock_invoke.call_count} {mock_invoke.call_args_list} {mock_invoke.call_args}")
        call_args = {**input_dict, 'print_prompt': "TEST", 'trained_state': model.trained_state, 'use_training': False}
        print(f"Expecting call {call_args}")
        mock_invoke.assert_called_once_with(**call_args)