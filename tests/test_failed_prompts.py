import pytest
from langdspy import PromptRunner, PromptSignature, InputField, OutputField, DefaultPromptStrategy
from langdspy.prompt_strategies import PromptStrategy
from langchain_contrib

class FailedPromptSignature(PromptSignature):
    input1 = InputField(name="input1", desc="Input field 1")
    input2 = InputField(name="input2", desc="Input field 2")
    output1 = OutputField(name="output1", desc="Output field 1")
    output2 = OutputField(name="output2", desc="Output field 2")

class FailedPromptStrategy(DefaultPromptStrategy):
    @staticmethod
    def _format_openai_prompt(trained_state, use_training, examples, **kwargs):
        # Simulate a prompt that doesn't include all required fields
        return "This is a simulated prompt without all required fields"

    @staticmethod
    def _format_anthropic_prompt(trained_state, use_training, examples, **kwargs):
        # Simulate a prompt that doesn't include all required fields
        return "This is a simulated prompt without all required fields"

    @staticmethod
    def _format_openai_json_prompt(trained_state, use_training, examples, **kwargs):
        # Simulate a prompt that doesn't include all required fields
        return '{"incomplete": "json"}'

    def parse_output_to_fields(self, output: str, llm_type: str) -> dict:
        # Simulate parsing that doesn't return all required fields
        return {"output1": "Some value"}

@pytest.mark.filterwarnings("ignore:.*cannot collect test class.*")
def test_failed_prompt_missing_input():
    prompt_runner = PromptRunner(template_class=FailedPromptSignature, prompt_strategy=FailedPromptStrategy)
    
    with pytest.raises(ValueError, match="Input keys do not match expected input keys"):
        prompt_runner.template.format_prompt(input1="test input", llm_type="openai")

@pytest.mark.filterwarnings("ignore:.*cannot collect test class.*")
def test_failed_prompt_extra_input():
    prompt_runner = PromptRunner(template_class=FailedPromptSignature, prompt_strategy=FailedPromptStrategy)
    
    with pytest.raises(ValueError, match="Input keys do not match expected input keys"):
        prompt_runner.template.format_prompt(input1="test input", input2="test input", extra_input="extra", llm_type="openai")

@pytest.mark.filterwarnings("ignore:.*cannot collect test class.*")
def test_failed_prompt_anthropic():
    prompt_runner = PromptRunner(template_class=FailedPromptSignature, prompt_strategy=FailedPromptStrategy)
    
    with pytest.raises(ValueError, match="Input keys do not match expected input keys"):
        prompt_runner.template.format_prompt(input1="test input", llm_type="anthropic")

@pytest.mark.filterwarnings("ignore:.*cannot collect test class.*")
def test_failed_prompt_openai_json():
    prompt_runner = PromptRunner(template_class=FailedPromptSignature, prompt_strategy=FailedPromptStrategy)
    
    with pytest.raises(ValueError, match="Input keys do not match expected input keys"):
        prompt_runner.template.format_prompt(input1="test input", llm_type="openai_json")

@pytest.mark.filterwarnings("ignore:.*cannot collect test class.*")
def test_failed_prompt_missing_output():
    prompt_runner = PromptRunner(template_class=FailedPromptSignature, prompt_strategy=FailedPromptStrategy)
    
    with pytest.raises(ValueError, match="Output keys do not match expected output keys"):
        result = prompt_runner.invoke({"input1": "test", "input2": "test"}, config={"llm_type": "openai"})