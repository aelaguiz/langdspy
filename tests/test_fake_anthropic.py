import pytest
from langdspy import PromptRunner, PromptSignature, InputField, OutputField, DefaultPromptStrategy, Model
from langchain_community.llms import FakeListLLM
# from langchain_contrib.llms.testing import FakeLLM

def create_test_model():
    class TestPromptSignature(PromptSignature):
        input = InputField(name="input", desc="Input field")
        output = OutputField(name="output", desc="Output field")


    class TestModel(Model):
        p1 = PromptRunner(template_class=TestPromptSignature, prompt_strategy=DefaultPromptStrategy)

        def invoke(self, input: str, config: dict) -> str:
            result = self.p1.invoke({"input": input}, config=config)
            return result

    return TestModel()

@pytest.fixture
def test_model():
    return create_test_model()


def test_fake_anthropic_llm(test_model):
    llm = FakeListLLM(verbose=True, responses=["<output>foo</output>", "<output>bar</output>", "<output>baz</output>"])
    result = test_model.invoke({"input": "test input"}, config={"llm": llm, "llm_type": "fake_anthropic"})
    assert result.output == "foo"

    result = test_model.invoke({"input": "test input"}, config={"llm": llm, "llm_type": "fake_anthropic"})
    assert result.output == "bar"

    result = test_model.invoke({"input": "test input"}, config={"llm": llm, "llm_type": "fake_anthropic"})
    assert result.output == "baz"