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
import logging

from field_descriptors import InputField, OutputField

logger = logging.getLogger(__name__)

class PromptSignature(BasePromptTemplate, BaseModel):
    # Assuming input_variables and output_variables are defined as class attributes
    input_variables: Dict[str, Any] = []
    output_variables: Dict[str, Any] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        inputs = {}
        outputs = {}

        for name, attribute in self.__class__.__fields__.items():
            if issubclass(attribute.type_, InputField):
                inputs[name] = attribute.default
            elif issubclass(attribute.type_, OutputField):
                outputs[name] = attribute.default

        self.input_variables = inputs
        self.output_variables = outputs

class PromptStrategy(BaseModel):
    def validate_inputs(self, inputs_dict):
        if not set(inputs_dict.keys()) == set(self.input_variables.keys()):
            logger.error(f"Input keys do not match expected input keys {inputs_dict.keys()} {self.input_variables.keys()}")
            raise ValueError(f"Input keys do not match expected input keys {inputs_dict.keys()} {self.input_variables.keys()}")

    def format(self, **kwargs: Any) -> str:
        return self.format_prompt(**kwargs)

    def _get_output_field(self, field_name):
        for output_name, output_field in self.output_variables.items():
            if output_field.name == field_name:
                return output_name




class DefaultPromptStrategy(PromptStrategy):
    def format_prompt(self, **kwargs: Any) -> str:
        # logger.debug(f"Formatting prompt with kwargs: {kwargs}")
        self.validate_inputs(kwargs)

        prompt = "Follow the following format. "

        if len(self.output_variables) > 1:
            prompt += "Fill any missing attributes and their values. Attributes that have values should not be changed or repeated."

        prompt += "\n\n"
        for input_name, input_field in self.input_variables.items():
            # prompt += f"⏎{input_field.name}: {input_field.desc}\n"
            prompt += input_field.format_prompt_description() + "\n"

        for output_name, output_field in self.output_variables.items():
            prompt += f"⏎{output_field.name}: {output_field.desc}\n"

        prompt += "\n---\n\n"

        for input_name, input_field in self.input_variables.items():
            prompt += input_field.format_prompt_value(kwargs.get(input_name)) + "\n"

        if len(self.output_variables) == 1:
            for output_name, output_field in self.output_variables.items():
                prompt += f"⏎{output_field.name}: \n"
        else:
            for output_name, output_field in self.output_variables.items():
                prompt += f"⏎{output_field.name}: \n"
                # prompt += f"\n"

        # logger.debug(f"Formatted prompt: {prompt}")
        print(prompt)
        return prompt

    def parse_output_to_fields(self, output: str) -> dict:
        try:
            pattern = r'^([^:]+): (.*)'
            lines = output.split('⏎')
            parsed_fields = {}

            logger.debug(f"Parsing output to fields with pattern {pattern} and lines {lines}")
            for line in lines:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    field_name, field_content = match.groups()
                    logger.debug(f"Matched line {line} - field name {field_name} field content {field_content}")
                    output_field = self._get_output_field(field_name)

                    if output_field:
                        parsed_fields[output_field] = field_content
                    else:
                        logger.error(f"Field {field_name} not found in output variables")
                else:
                    logger.debug(f"NO MATCH line {line}")
                    
            if not parsed_fields and len(self.output_variables) == 1:
                parsed_fields[list(self.output_variables.keys())[0]] = line


            return parsed_fields
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            raise e



class ChainOfThought(PromptStrategy):
    """

    """
    def format_prompt(self, **kwargs: Any) -> str:
        self.validate_inputs(kwargs)

        ## IMPLEMENT THIS
        
        return prompt

    def parse_output_to_fields(self, output: str) -> dict:
        """
        Parses the provided output string into a dictionary with keys as field names 
        and values as the corresponding field contents, using regex.

        Parameters:
        output (str): The string output to be parsed.

        Returns:
        dict: A dictionary where each key is an output field name and each value is the content of that field.
        """

        ### IMPLEMENT THIS

        return parsed_fields