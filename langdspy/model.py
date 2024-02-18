from langchain.prompts import BasePromptTemplate  # Assuming this is the correct import path
import re
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
    def __init__(self, name:str, desc: str, formatter: Optional[Callable[[Any], Any]] = None, transformer: Optional[Callable[[Any], Any]] = None, validator: Optional[Callable[[Any], Any]] = None):
        assert "⏎" not in name, "Field name cannot contain newline character"
        assert ":" not in name, "Field name cannot contain colon character"

        self.name = name
        self.desc = desc
        self.formatter = formatter
        self.transformer = transformer
        self.validator = validator


    def format_value(self, value: Any) -> Any:
        if self.formatter:
            return self.formatter(value)
        else:
            return value

    def transform_value(self, value: Any) -> Any:
        if self.transformer:
            return self.transformer(value)
        else:
            return value

    def validate_value(self, input: Input, value: Any) -> bool:
        if self.validator:
            return self.validator(input, value)
        else:
            return True

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

class PromptStrategy(BaseModel):
    # def generate_prediction_return(self, **kwargs, output) -> Prediction:
    #     pass
    pass



class DefaultPromptStrategy(PromptStrategy):
    def format_prompt(self, **kwargs: Any) -> str:
        self.validate_inputs(kwargs)

        prompt = "Follow the following format. "

        if len(self.output_variables) > 1:
            prompt += "Fill any missing attributes and their values."

        prompt += "\n\n"
        for input_name, input_field in self.input_variables.items():
            prompt += f"⏎{input_field.name}: {input_field.desc}\n"

        for output_name, output_field in self.output_variables.items():
            prompt += f"⏎{output_field.name}: {output_field.desc}\n"

        prompt += "\n---\n\n"

        for input_name, input_field in self.input_variables.items():
            val = input_field.format_value(kwargs.get(input_name))

            prompt += f"⏎{input_field.name}: {val}\n"

        if len(self.output_variables) == 1:
            for output_name, output_field in self.output_variables.items():
                prompt += f"⏎{output_field.name}: \n"
        else:
            for output_name, output_field in self.output_variables.items():
                prompt += f"\n"

        logger.debug(f"Formatted prompt: {prompt}")
        return prompt

    def _get_output_field(self, field_name):
        for output_name, output_field in self.output_variables.items():
            if output_field.name == field_name:
                return output_name

        assert False, f"Field {field_name} not found in output variables"

    def parse_output_to_fields(self, output: str) -> dict:
        """
        Parses the provided output string into a dictionary with keys as field names 
        and values as the corresponding field contents, using regex.

        Parameters:
        output (str): The string output to be parsed.

        Returns:
        dict: A dictionary where each key is an output field name and each value is the content of that field.
        """
        # Regular expression pattern to match field name and content
        pattern = r'([^:]+):(.*)'

        # Split the output by the special character
        lines = output.split('⏎')

        # Dictionary to hold the parsed fields
        parsed_fields = {}

        for line in lines:
            # If there are multiple outputs we should assume that we'll get the labels with each, if just one we'll just assign the whole thing
            if len(self.output_variables) > 1:
                match = re.match(pattern, line)
                if match:
                    # Extract field name and content from the match
                    field_name, field_content = match.groups()

                    output_field = self._get_output_field(field_name)
                    parsed_fields[output_field] = field_content
            else:
                parsed_fields[list(self.output_variables.keys())[0]] = line

        return parsed_fields

    def validate_inputs(self, inputs_dict):
        assert set(inputs_dict.keys()) == set(self.input_variables.keys()), "Input keys do not match expected input keys"

    def format(self, **kwargs: Any) -> str:
        return self.format_prompt(**kwargs)
        
class PromptRunner(RunnableSerializable):
    template: PromptSignature = None

    def __init__(self, template_class, prompt_strategy):
        super().__init__()
        cls_ = type(template_class.__name__, (prompt_strategy, template_class), {})
        self.template = cls_()
    
    @validator("template")
    def check_template(
        cls, value: PromptSignature
    ) -> PromptSignature:
        return value
    
    def _invoke_with_retries(self, chain, input, max_tries=1, config: Optional[RunnableConfig] = {}):
        total_max_tries = max_tries

        hard_fail = config.get('hard_fail', False)

        res = {}

        while max_tries >= 1:
            res = chain.invoke(input, config=config)
            validation = True

            logger.debug(f"Raw output for prompt runner {self.template.__class__.__name__}: {res}")

            # Use the parse_output_to_fields method from the PromptStrategy
            parsed_output = self.template.parse_output_to_fields(res)
            logger.debug(f"Parsed output: {parsed_output}")

            len_parsed_output = len(parsed_output.keys())
            len_output_variables = len(self.template.output_variables.keys())
            logger.debug(f"Parsed output keys: {parsed_output.keys()} [{len_parsed_output}] Expected output keys: {self.template.output_variables.keys()} [{len_output_variables}]")

            if len(parsed_output.keys()) != len(self.template.output_variables.keys()):
                logger.error(f"Output keys do not match expected output keys for prompt runner {self.template.__class__.__name__}")
                validation = False

            if validation:
                # Transform and validate the outputs
                for attr_name, output_field in self.template.output_variables.items():
                    output_value = parsed_output.get(attr_name)
                    if not output_value:
                        logger.error(f"Failed to get output value for field {attr_name} for prompt runner {self.template.__class__.__name__}")
                        validation = False
                        continue

                    # Get the transformed value
                    transformed_val = output_field.transform_value(output_value)

                    # Validate the transformed value
                    if not output_field.validate_value(input, transformed_val):
                        validation = False
                        logger.error(f"Failed to validate field {attr_name} value {transformed_val} for prompt runner {self.template.__class__.__name__}")

                    # Update the output with the transformed value
                    parsed_output[attr_name] = transformed_val

            res = {attr_name: parsed_output[attr_name] for attr_name in self.template.output_variables.keys()}
            if validation:
                # Return a dictionary keyed by attribute names with validated values
                return res

            logger.error(f"Output validation failed for prompt runner {self.template.__class__.__name__}")
            max_tries -= 1

        if hard_fail:
            raise ValueError(f"Output validation failed for prompt runner {self.template.__class__.__name__} after {total_max_tries} tries.")

        return res
 
    def invoke(self, input: Input, config: Optional[RunnableConfig] = {}) -> Output:
        prompt = self.template.format(**input)

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

        prediction_data = {**input, **res}


        prediction = Prediction(**prediction_data)

        return prediction

class Model(RunnableSerializable):
    prompt_runners = []

    def __init__(self):
        super().__init__()

        for field_name, field in self.__fields__.items():
            if issubclass(field.type_, PromptRunner):
                self.prompt_runners.append((field_name, field.default))
