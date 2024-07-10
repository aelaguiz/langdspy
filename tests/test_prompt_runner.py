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
from langdspy import PromptRunner, DefaultPromptStrategy, InputField, OutputField, Model, PromptSignature, Prediction, PromptStrategy

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


from langchain.chat_models.base import BaseChatModel

class TestLLM(BaseChatModel):
    def invoke(self, *args, **kwargs):
        return "INVOKED"

    def _generate(self, *args, **kwargs):
        return None

    def _llm_type(self) -> str:
        return "test"

def test_trained_state_in_inputs():
    model = TestModel(n_jobs=1)
    input_dict = {"input": "Test input"}
    mock_invoke = MagicMock(return_value="FORMATTED PROMPT")
    
    with patch.object(PromptStrategy, 'format_prompt', new=mock_invoke):
        config = {"llm": TestLLM(), "llm_type": "test"}
        model.trained_state.examples = [("EXAMPLE_X", "EXAMPLE_Y")]
        result = model.invoke(input_dict, config=config)

        print(result)
        print(f"Called with {mock_invoke.call_count} {mock_invoke.call_args_list} {mock_invoke.call_args}")
        call_args = {**input_dict, 'trained_state': model.trained_state, 'use_training': True,  'llm_type': "test"}
        print(f"Expecting call {call_args}")
        mock_invoke.assert_called_with(**call_args)

def test_use_training():
    model = TestModel(n_jobs=1)
    input_dict = {"input": "Test input"}
    mock_invoke = MagicMock(return_value="FORMATTED PROMPT")
    
    with patch.object(DefaultPromptStrategy, 'format_prompt', new=mock_invoke):
        config = {"llm": TestLLM(), "use_training": False, "llm_type": "test"}
        model.trained_state.examples = [("EXAMPLE_X", "EXAMPLE_Y")]
        result = model.invoke(input_dict, config=config)

        print(result)
        print(f"Called with {mock_invoke.call_count} {mock_invoke.call_args_list} {mock_invoke.call_args}")
        call_args = {**input_dict, 'trained_state': model.trained_state, 'use_training': False,  'llm_type': "test"}
        print(f"Expecting call {call_args}")
        mock_invoke.assert_called_with(**call_args)