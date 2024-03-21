# tests/test_prompt_history.py
import pytest
from langdspy import Model, PromptRunner, PromptSignature, InputField, OutputField, DefaultPromptStrategy

class TestPromptSignature1(PromptSignature):
    input1 = InputField(name="input1", desc="Input field 1")
    output1 = OutputField(name="output1", desc="Output field 1")

class TestPromptSignature2(PromptSignature):
    input2 = InputField(name="input2", desc="Input field 2")
    output2 = OutputField(name="output2", desc="Output field 2")

@pytest.fixture
def llm():
    class FakeLLM:
        model = "test one"

        def __call__(self, prompt, stop=None):
            return "Fake LLM response"
    return FakeLLM()

def test_prompt_history(llm):
    class TestModel(Model):
        prompt_runner1 = PromptRunner(template_class=TestPromptSignature1, prompt_strategy=DefaultPromptStrategy)
        prompt_runner2 = PromptRunner(template_class=TestPromptSignature2, prompt_strategy=DefaultPromptStrategy)

        def invoke(self, input_dict, config):
            result1 = self.prompt_runner1.invoke({"input1": input_dict["input1"]}, config=config)
            result2 = self.prompt_runner2.invoke({"input2": input_dict["input2"]}, config=config)
            return {"output1": result1.output1, "output2": result2.output2}

    model = TestModel(n_jobs=1)

    input_dict = {"input1": "Test input 1", "input2": "Test input 2"}
    config = {"llm": llm}

    result = model.invoke(input_dict, config=config)

    assert result["output1"] == "Fake LLM response"
    assert result["output2"] == "Fake LLM response"

    prompt_history = model.get_prompt_history()
    assert len(prompt_history) == 2

    runner_name1, entry1 = prompt_history[0]
    assert runner_name1 == "prompt_runner1"
    assert "prompt" in entry1
    assert "llm" in entry1
    assert "llm_response" in entry1
    assert entry1["llm"] == "openai test one"
    assert entry1["parsed_output"] == {"output1": "Fake LLM response"}
    assert entry1["error"] is None
    assert "duration_ms" in entry1
    assert "timestamp" in entry1

    runner_name2, entry2 = prompt_history[1]
    assert runner_name2 == "prompt_runner2"
    assert "prompt" in entry2
    assert "llm" in entry2
    assert "llm_response" in entry2
    assert entry2["llm"] == "openai test one"
    assert entry2["parsed_output"] == {"output2": "Fake LLM response"}
    assert entry2["error"] is None
    assert "duration_ms" in entry2
    assert "timestamp" in entry2

def test_failed_prompts(llm):
    class TestModel(Model):
        prompt_runner1 = PromptRunner(template_class=TestPromptSignature1, prompt_strategy=DefaultPromptStrategy)
        prompt_runner2 = PromptRunner(template_class=TestPromptSignature2, prompt_strategy=DefaultPromptStrategy)

        def invoke(self, input_dict, config):
            result1 = self.prompt_runner1.invoke({"input1": input_dict["input1"]}, config=config)
            result2 = self.prompt_runner2.invoke({"input2": input_dict["input2"]}, config=config)
            return {"output1": result1.output1, "output2": result2.output2}
    model = TestModel(n_jobs=1)
    input_dict = {"input1": "Test input 1", "input2": "Test input 2"}
    config = {"llm": llm}

    result = model.invoke(input_dict, config=config)

    failed_prompts = model.get_failed_prompts()
    print(failed_prompts)
    assert len(failed_prompts) == 0

    successful_prompts = model.get_successful_prompts()
    print(successful_prompts)
    assert len(successful_prompts) == 2

    runner_name1, entry1 = successful_prompts[0]
    assert runner_name1 == "prompt_runner1"
    assert entry1["error"] is None

    runner_name2, entry2 = successful_prompts[1]
    assert runner_name2 == "prompt_runner2"
    assert entry2["error"] is None
