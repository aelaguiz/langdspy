# tests/test_prompt_formatting.py
import pytest
from langdspy.field_descriptors import InputField, OutputField, HintField
from langdspy.prompt_strategies import PromptSignature, DefaultPromptStrategy
from langdspy.prompt_runners import PromptRunner
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

class TestPromptSignature(PromptSignature):
    input = InputField(name="input", desc="Input field")
    output = OutputField(name="output", desc="Output field")
    hint = HintField(desc="Hint field")

def test_format_prompt_openai():
    prompt_runner = PromptRunner(template_class=TestPromptSignature, prompt_strategy=DefaultPromptStrategy)
    
    formatted_prompt = prompt_runner.template._format_openai_prompt(trained_state=None, use_training=True, input="test input")
    print(formatted_prompt)
    
    assert "💡 Hint field" in formatted_prompt
    assert "✅input: Input field" in formatted_prompt
    assert "🔑output: Output field" in formatted_prompt
    assert "✅input: test input" in formatted_prompt
    assert "🔑output:" in formatted_prompt

def test_format_prompt_anthropic():
    prompt_runner = PromptRunner(template_class=TestPromptSignature, prompt_strategy=DefaultPromptStrategy)
    
    formatted_prompt = prompt_runner.template._format_anthropic_prompt(trained_state=None, use_training=True, input="test input")
    
    assert "<hint>Hint field</hint>" in formatted_prompt
    assert "<input>: Input field" in formatted_prompt
    assert "<output>: Output field" in formatted_prompt
    assert "<input>test input</input>" in formatted_prompt
    assert "<output></output>" in formatted_prompt

def test_parse_output_openai():
    prompt_runner = PromptRunner(template_class=TestPromptSignature, prompt_strategy=DefaultPromptStrategy)
    
    output = "🔑output: test output"
    parsed_output = prompt_runner.template._parse_openai_output_to_fields(output)
    
    assert parsed_output["output"] == "test output"

def test_parse_output_anthropic():
    prompt_runner = PromptRunner(template_class=TestPromptSignature, prompt_strategy=DefaultPromptStrategy)
    
    output = "<output>test output</output>"
    parsed_output = prompt_runner.template._parse_anthropic_output_to_fields(output)
    
    assert parsed_output["output"] == "test output"

def test_llm_type_detection_openai():
    prompt_runner = PromptRunner(template_class=TestPromptSignature, prompt_strategy=DefaultPromptStrategy)
    
    llm = ChatOpenAI()
    llm_type = prompt_runner._determine_llm_type(llm)
    
    assert llm_type == "openai"

def test_llm_type_detection_anthropic():
    prompt_runner = PromptRunner(template_class=TestPromptSignature, prompt_strategy=DefaultPromptStrategy)
    
    llm = ChatAnthropic(model_name="claude-3-sonnet-20240229")
    llm_type = prompt_runner._determine_llm_type(llm)
    
    assert llm_type == "anthropic"