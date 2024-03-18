import pytest
from langdspy.field_descriptors import InputField, OutputField, HintField
from langdspy.prompt_strategies import PromptSignature, DefaultPromptStrategy
from langdspy.prompt_runners import PromptRunner

class TestPromptSignature(PromptSignature):
    input = InputField(name="input", desc="Input field")
    output = OutputField(name="output", desc="Output field")
    hint = HintField(desc="Hint field")
    __examples__ = [
        ({"input": "Example input 1"}, "Example output 1"),
        ({"input": "Example input 2"}, "Example output 2"),
    ]

def test_format_prompt_with_examples_openai():
    prompt_runner = PromptRunner(template_class=TestPromptSignature, prompt_strategy=DefaultPromptStrategy)
    formatted_prompt = prompt_runner.template._format_openai_prompt(
        trained_state=None,
        use_training=True,
        examples=TestPromptSignature.__examples__,
        input="Test input"
    )
    print(formatted_prompt)
    assert "💡 Hint field" in formatted_prompt
    assert "✅input: Input field" in formatted_prompt
    assert "🔑output: Output field" in formatted_prompt
    assert "✅input: Example input 1" in formatted_prompt
    assert "🔑output: Example output 1" in formatted_prompt
    assert "✅input: Example input 2" in formatted_prompt
    assert "🔑output: Example output 2" in formatted_prompt
    assert "✅input: Test input" in formatted_prompt
    assert "🔑output:" in formatted_prompt

def test_format_prompt_with_examples_anthropic():
    prompt_runner = PromptRunner(template_class=TestPromptSignature, prompt_strategy=DefaultPromptStrategy)
    formatted_prompt = prompt_runner.template._format_anthropic_prompt(
        trained_state=None,
        use_training=True,
        examples=TestPromptSignature.__examples__,
        input="Test input"
    )
    assert "<hint>Hint field</hint>" in formatted_prompt
    assert "<input>Input field</input>" in formatted_prompt
    assert "<output>Output field</output>" in formatted_prompt
    assert "<input>Example input 1</input>" in formatted_prompt
    assert "<output>Example output 1</output>" in formatted_prompt
    assert "<input>Example input 2</input>" in formatted_prompt
    assert "<output>Example output 2</output>" in formatted_prompt
    assert "<input>Test input</input>" in formatted_prompt
    assert "<output></output>" in formatted_prompt

def test_format_prompt_without_examples_openai():
    prompt_runner = PromptRunner(template_class=TestPromptSignature, prompt_strategy=DefaultPromptStrategy)
    formatted_prompt = prompt_runner.template._format_openai_prompt(
        trained_state=None,
        use_training=False,
        examples=[],
        input="Test input"
    )
    assert "💡 Hint field" in formatted_prompt
    assert "✅input: Input field" in formatted_prompt
    assert "🔑output: Output field" in formatted_prompt
    assert "✅input: Test input" in formatted_prompt
    assert "🔑output:" in formatted_prompt
    assert "Example input 1" not in formatted_prompt
    assert "Example output 1" not in formatted_prompt
    assert "Example input 2" not in formatted_prompt
    assert "Example output 2" not in formatted_prompt

def test_format_prompt_without_examples_anthropic():
    prompt_runner = PromptRunner(template_class=TestPromptSignature, prompt_strategy=DefaultPromptStrategy)
    formatted_prompt = prompt_runner.template._format_anthropic_prompt(
        trained_state=None,
        use_training=False,
        examples=[],
        input="Test input"
    )
    assert "<hint>Hint field</hint>" in formatted_prompt
    assert "<input>Input field</input>" in formatted_prompt
    assert "<output>Output field</output>" in formatted_prompt
    assert "<input>Test input</input>" in formatted_prompt
    assert "<output></output>" in formatted_prompt
    assert "Example input 1" not in formatted_prompt
    assert "Example output 1" not in formatted_prompt
    assert "Example input 2" not in formatted_prompt
    assert "Example output 2" not in formatted_prompt

def test_validate_examples_valid():
    class ValidPromptSignature(PromptSignature):
        input1 = InputField(name="input1", desc="Input field 1")
        input2 = InputField(name="input2", desc="Input field 2")
        output1 = OutputField(name="output1", desc="Output field 1")
        output2 = OutputField(name="output2", desc="Output field 2")
        __examples__ = [
            ({"input1": "Example input 1", "input2": "Example input 2"}, {"output1": "Example output 1", "output2": "Example output 2"}),
        ]
    prompt_runner = PromptRunner(template_class=ValidPromptSignature, prompt_strategy=DefaultPromptStrategy)
    prompt_runner.template.validate_examples()  # Should not raise any exception

def test_validate_examples_invalid_input_field():
    class InvalidInputPromptSignature(PromptSignature):
        input = InputField(name="input", desc="Input field")
        output = OutputField(name="output", desc="Output field")
        __examples__ = [
            ({"invalid_input": "Example input"}, "Example output"),
        ]
    with pytest.raises(ValueError, match="Example input field 'invalid_input' not found in input_variables"):
        PromptRunner(template_class=InvalidInputPromptSignature, prompt_strategy=DefaultPromptStrategy)

def test_validate_examples_invalid_output_field():
    class InvalidOutputPromptSignature(PromptSignature):
        input = InputField(name="input", desc="Input field")
        output = OutputField(name="output", desc="Output field")
        __examples__ = [
            ({"input": "Example input"}, {"invalid_output": "Example output"}),
        ]
    with pytest.raises(ValueError, match="Example output field 'invalid_output' not found in output_variables"):
        PromptRunner(template_class=InvalidOutputPromptSignature, prompt_strategy=DefaultPromptStrategy)

def test_validate_examples_invalid_output_format():
    class InvalidOutputFormatPromptSignature(PromptSignature):
        input = InputField(name="input", desc="Input field")
        output1 = OutputField(name="output1", desc="Output field 1")
        output2 = OutputField(name="output2", desc="Output field 2")
        __examples__ = [
            ({"input": "Example input"}, "Example output"),
        ]
    with pytest.raises(ValueError, match="Example output must be a dictionary when there are multiple output fields"):
        PromptRunner(template_class=InvalidOutputFormatPromptSignature, prompt_strategy=DefaultPromptStrategy)

def test_format_prompt_with_multiple_output_fields_openai():
    class MultipleOutputPromptSignature(PromptSignature):
        input = InputField(name="input", desc="Input field")
        output1 = OutputField(name="output1", desc="Output field 1")
        output2 = OutputField(name="output2", desc="Output field 2")
        __examples__ = [
            ({"input": "Example input"}, {"output1": "Example output 1", "output2": "Example output 2"}),
        ]
    prompt_runner = PromptRunner(template_class=MultipleOutputPromptSignature, prompt_strategy=DefaultPromptStrategy)
    formatted_prompt = prompt_runner.template._format_openai_prompt(
        trained_state=None,
        use_training=True,
        examples=MultipleOutputPromptSignature.__examples__,
        input="Test input"
    )
    assert "✅input: Example input" in formatted_prompt
    assert "🔑output1: Example output 1" in formatted_prompt
    assert "🔑output2: Example output 2" in formatted_prompt

def test_format_prompt_with_multiple_output_fields_anthropic():
    class MultipleOutputPromptSignature(PromptSignature):
        input = InputField(name="input", desc="Input field")
        output1 = OutputField(name="output1", desc="Output field 1")
        output2 = OutputField(name="output2", desc="Output field 2")
        __examples__ = [
            ({"input": "Example input"}, {"output1": "Example output 1", "output2": "Example output 2"}),
        ]
    prompt_runner = PromptRunner(template_class=MultipleOutputPromptSignature, prompt_strategy=DefaultPromptStrategy)
    formatted_prompt = prompt_runner.template._format_anthropic_prompt(
        trained_state=None,
        use_training=True,
        examples=MultipleOutputPromptSignature.__examples__,
        input="Test input"
    )
    assert "<input>Example input</input>" in formatted_prompt
    assert "<output1>Example output 1</output1>" in formatted_prompt
    assert "<output2>Example output 2</output2>" in formatted_prompt