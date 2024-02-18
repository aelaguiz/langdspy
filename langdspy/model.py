from langchain.prompts import BasePromptTemplate  # Assuming this is the correct import path
from langchain.prompts import FewShotPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator, Extra
from langchain_core.pydantic_v1 import validator
from langchain_core.language_models import BaseLLM
from typing import Any, Dict, List, Type, Optional, Callable
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_core.runnables.utils import (
    Input,
    Output
)
from langchain_core.runnables.config import (
    RunnableConfig
)
from . import lcel_logger

import logging

logger = logging.getLogger(__name__)

class FieldDescriptor:
    def __init__(self, name:str, desc: str, formatter: Optional[Callable[[Any], Any]] = None):
        self.name = name
        self.desc = desc
        self.formatter = formatter


    def format_value(self, value: Any) -> Any:
        if self.formatter:
            return self.formatter(value)
        else:
            return value

class InputField(FieldDescriptor):
    pass

class OutputField(FieldDescriptor):
    pass

class Prediction(BaseModel):
    class Config:
        extra = Extra.allow  # This allows the model to accept extra fields that are not explicitly declared

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Initialize BaseModel with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)  # Dynamically assign attributes

class PromptSignature(BasePromptTemplate, BaseModel):
    # Assuming input_variables and output_variables are defined as class attributes
    input_variables: Dict[str, Any] = []
    output_variables: Dict[str, Any] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        inputs = {}
        outputs = {}

        for name, attribute in self.__class__.__fields__.items():
            if attribute.type_ == InputField:
                inputs[name] = attribute.default
            elif attribute.type_ == OutputField:
                outputs[name] = attribute.default

        self.input_variables = inputs
        self.output_variables = outputs

    def validate_output(self, output: Output) -> bool:
        # Default implementation, can be overridden
        return True


class PromptStrategy(BaseModel):
    # def generate_prediction_return(self, **kwargs, output) -> Prediction:
    #     pass
    pass



class DefaultPromptStrategy(PromptStrategy):
    def format_prompt(self, **kwargs: Any) -> str:
        self.validate_inputs(kwargs)

        print("Formatting prompt")
        print(f"Input variables: self.input_variables: {self.input_variables}")

        prompt = "Follow the following format.\n\n"
        for input_name, input_field in self.input_variables.items():
            prompt += f"{input_field.name}: {input_field.desc}\n"

        for output_name, output_field in self.output_variables.items():
            prompt += f"{output_field.name}: {output_field.desc}\n"

        prompt += "\n---\n\n"

        for input_name, input_field in self.input_variables.items():
            val = input_field.format_value(kwargs.get(input_name))

            prompt += f"{input_field.name}: {val}\n"

        prompt += f"{output_field.name}: "

        return prompt

    def validate_inputs(self, inputs_dict):
        assert set(inputs_dict.keys()) == set(self.input_variables.keys()), "Input keys do not match expected input keys"

    def format(self, **kwargs: Any) -> str:
        return self.format_prompt(**kwargs)
        
class PromptRunner(RunnableSerializable):
    template: PromptSignature = None

    def __init__(self, template_class, prompt_strategy):
        print(f"Initializing prompt runner")
        super().__init__()
        cls_ = type(template_class.__name__, (prompt_strategy, template_class), {})
        self.template = cls_()
    
    @validator("template")
    def check_template(
        cls, value: PromptSignature
    ) -> PromptSignature:
        print(f"Checking template: {value}")
        return value
    
    def _invoke_with_retries(self, chain, input, max_tries = 1, config: Optional[RunnableConfig] = None):
        while max_tries >= 1:
            res = chain.invoke(input, config={'callbacks': [lcel_logger.LlmDebugHandler()]})

            logger.debug(f"Validating output for prompt runner {self.template.__class__.__name__}: {res}")
            if self.template.validate_output(res):
                return res

            logger.error(f"Output validation failed for prompt runner {self.template.__class__.__name__}")
            max_tries -= 1

        raise ValueError(f"Output validation failed for prompt runner {self.template.__class__.__name__}")
        
    def invoke(self, input: Input, config: Optional[RunnableConfig] = {}) -> Output:
        print(f"Prompt runner {self.template.__class__.__name__} invoked with input: {input}")
        prompt = self.template.format(**input)
        print(f"Prompt: {prompt}")

        logger.debug(f"Template: {self.template}")
        logger.debug(f"Config: {config}")
        chain = (
            self.template
            | config['llm']
            | StrOutputParser()
        )

        max_retries = config.get('max_tries', 3)

        res = self._invoke_with_retries(chain, input, max_retries, config=config)

        logger.debug(f"Result: {res}")

        prediction_data = {**input}

        for output_var_name in self.template.output_variables.keys():
            prediction_data[output_var_name] = res

        prediction = Prediction(**prediction_data)

        return prediction

class Model(RunnableSerializable):
    prompt_runners = []

    def __init__(self):
        super().__init__()

        for field_name, field in self.__fields__.items():
            if issubclass(field.type_, PromptRunner):
                self.prompt_runners.append((field_name, field.default))
