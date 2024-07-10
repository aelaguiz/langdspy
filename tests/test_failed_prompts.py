import pytest
from langdspy import PromptRunner, PromptSignature, InputField, OutputField, DefaultPromptStrategy, Model
from langchain_community.llms import FakeListLLM

class FailedPromptSignature(PromptSignature):
    input1 = InputField(name="input1", desc="Input field 1")
    input2 = InputField(name="input2", desc="Input field 2")
    output1 = OutputField(name="output1", desc="Output field 1")
    output2 = OutputField(name="output2", desc="Output field 2")

class OptionalOutputPromptSignature(PromptSignature):
    input1 = InputField(name="input1", desc="Input field 1")
    input2 = InputField(name="input2", desc="Input field 2")
    output1 = OutputField(name="output1", desc="Output field 1")
    output2 = OutputField(name="output2", desc="Output field 2", optional=True)

class FailedModel(Model):
    failed_prompt = PromptRunner(template_class=FailedPromptSignature, prompt_strategy=DefaultPromptStrategy)

    def invoke(self, input: dict, config: dict) -> dict:
        result = self.failed_prompt.invoke(input, config=config)
        return {"output1": result.output1, "output2": result.output2}

class OptionalOutputModel(Model):
    optional_prompt = PromptRunner(template_class=OptionalOutputPromptSignature, prompt_strategy=DefaultPromptStrategy)

    def invoke(self, input: dict, config: dict) -> dict:
        result = self.optional_prompt.invoke(input, config=config)
        return {"output1": result.output1, "output2": result.output2}

@pytest.fixture
def failed_model():
    return FailedModel()

@pytest.fixture
def optional_output_model():
    return OptionalOutputModel()

def test_failed_prompt_missing_output(failed_model):
    llm = FakeListLLM(responses=["<output1>Some value</output1>"])
    
    with pytest.raises(ValueError, match="Output validation failed for prompt runner"):
        failed_model.invoke({"input1": "test", "input2": "test"}, config={"llm": llm, "llm_type": "fake_anthropic"})

def test_optional_output_field(optional_output_model):
    llm = FakeListLLM(responses=["<output1>Some value</output1>"])
    
    result = optional_output_model.invoke({"input1": "test", "input2": "test"}, config={"llm": llm, "llm_type": "fake_anthropic"})
    
    assert result["output1"] == "Some value"
    assert result["output2"] is None

# def test_failed_prompt_invalid_output(failed_model):
#     llm = FakeListLLM(responses=["<output1>Some value</output1><output2>Invalid value</output2>"])
    
#     result = failed_model.invoke({"input1": "test", "input2": "test"}, config={"llm": llm, "llm_type": "fake_anthropic"})
    
#     failed_prompts = failed_model.get_failed_prompts()
#     assert len(failed_prompts) == 1
#     assert failed_prompts[0][1]["error"] is not None

# def test_successful_prompt(failed_model):
#     llm = FakeListLLM(responses=["<output1>Valid value 1</output1><output2>Valid value 2</output2>"])
    
#     result = failed_model.invoke({"input1": "test", "input2": "test"}, config={"llm": llm, "llm_type": "fake_anthropic"})
    
#     assert result["output1"] == "Valid value 1"
#     assert result["output2"] == "Valid value 2"
    
#     successful_prompts = failed_model.get_successful_prompts()
#     assert len(successful_prompts) == 1
#     assert successful_prompts[0][1]["error"] is None
