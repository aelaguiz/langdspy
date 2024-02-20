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
        logger.debug(f"Input keys to validate: {inputs_dict.keys()} {self.input_variables.keys()}")
        assert set(inputs_dict.keys()) == set(self.input_variables.keys()), "Input keys do not match expected input keys"

    def format(self, **kwargs: Any) -> str:
        return self.format_prompt(**kwargs)

    def _get_output_field(self, field_name):
        for output_name, output_field in self.output_variables.items():
            if output_field.name == field_name:
                return output_name

        assert False, f"Field {field_name} not found in output variables"




class DefaultPromptStrategy(PromptStrategy):
    def format_prompt(self, **kwargs: Any) -> str:
        # logger.debug(f"Formatting prompt with kwargs: {kwargs}")
        self.validate_inputs(kwargs)

        prompt = "Follow the following format. "

        if len(self.output_variables) > 1:
            prompt += "Fill any missing attributes and their values. Do not repeat attributes that have values present."

        prompt += "\n\n"
        for input_name, input_field in self.input_variables.items():
            # prompt += f"⏎{input_field.name}: {input_field.desc}\n"
            prompt += input_field.format_prompt() + "\n"

        for output_name, output_field in self.output_variables.items():
            prompt += f"⏎{output_field.name}: {output_field.desc}\n"

        prompt += "\n---\n\n"

        for input_name, input_field in self.input_variables.items():
            prompt += input_field.format_prompt(kwargs.get(input_name)) + "\n"

        if len(self.output_variables) == 1:
            for output_name, output_field in self.output_variables.items():
                prompt += f"⏎{output_field.name}: \n"
        else:
            for output_name, output_field in self.output_variables.items():
                prompt += f"⏎{output_field.name}: \n"
                # prompt += f"\n"

        logger.debug(f"Formatted prompt: {prompt}")
        print(prompt)
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