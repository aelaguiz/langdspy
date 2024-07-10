import pytest
from langdspy import PromptRunner, PromptSignature, InputField, OutputField, DefaultPromptStrategy
from langchain_contrib.llms.testing import FakeLLM

class TestPromptSignature(PromptSignature):
    input = InputField(name="input", desc="Input field")
    output = OutputField(name="output", desc="Output field")

def test_fake_anthropic_llm():
    prompt_runner = PromptRunner(template_class=TestPromptSignature, prompt_strategy=DefaultPromptStrategy)
    
    # Test with default response
    llm = FakeLLM()
    result = prompt_runner.invoke({"input": "test input"}, config={"llm": llm, "llm_type": "fake_anthropic"})
    assert result.output == "foo"

    # Test with custom mapped response
    llm = FakeLLM(mapped_responses={"<input>test input</input>": "<output>Custom response</output>"})
    result = prompt_runner.invoke({"input": "test input"}, config={"llm": llm, "llm_type": "fake_anthropic"})
    assert result.output == "Custom response"

    # Test with sequenced responses
    llm = FakeLLM(sequenced_responses=["<output>One</output>", "<output>Two</output>", "<output>Three</output>"])
    
    result1 = prompt_runner.invoke({"input": "test input"}, config={"llm": llm, "llm_type": "fake_anthropic"})
    assert result1.output == "One"
    
    result2 = prompt_runner.invoke({"input": "test input"}, config={"llm": llm, "llm_type": "fake_anthropic"})
    assert result2.output == "Two"
    
    result3 = prompt_runner.invoke({"input": "test input"}, config={"llm": llm, "llm_type": "fake_anthropic"})
    assert result3.output == "Three"
    
    result4 = prompt_runner.invoke({"input": "test input"}, config={"llm": llm, "llm_type": "fake_anthropic"})
    assert result4.output == "foo"  # Default response after exhausting sequenced responses
