from langchain.prompts import BasePromptTemplate  # Assuming this is the correct import path
import json
import re
from langchain.prompts import FewShotPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator, Extra
from langchain_core.pydantic_v1 import validator
from langchain_core.language_models import BaseLLM
from typing import Any, Dict, List, Type, Optional, Callable, Tuple
import uuid
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

from .field_descriptors import InputField, OutputField, HintField

logger = logging.getLogger("langdspy")

class PromptSignature(BasePromptTemplate, BaseModel):
    input_variables: Dict[str, Any] = []
    output_variables: Dict[str, Any] = []
    hint_variables: Dict[str, Any] = []  # New attribute for hint fields
    instance_id: str = Field(default_factory=str)
    __examples__: List[Tuple[Dict[str, Any], Any]] = []


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.instance_id = str(uuid.uuid4())  # Generate a unique identifier


        inputs = {}
        outputs = {}
        hints = {}  # New dictionary to hold hint fields

        for name, attribute in self.__class__.__fields__.items():
            if issubclass(attribute.type_, InputField):
                inputs[name] = attribute.default
            elif issubclass(attribute.type_, OutputField):
                outputs[name] = attribute.default
            elif issubclass(attribute.type_, HintField):  # Check if the field is a HintField
                hints[name] = attribute.default 

        self.input_variables = inputs
        self.output_variables = outputs
        self.hint_variables = hints 

        self.validate_examples()

    def validate_examples(self):
        for example_input, example_output in self.__examples__:
            # Check input fields
            for input_name in example_input:
                if input_name not in self.input_variables:
                    raise ValueError(f"Example input field '{input_name}' not found in input_variables")

            # Check output fields
            if isinstance(example_output, dict):
                for output_name in example_output:
                    if output_name not in self.output_variables:
                        raise ValueError(f"Example output field '{output_name}' not found in output_variables")
            else:
                if len(self.output_variables) != 1:
                    raise ValueError("Example output must be a dictionary when there are multiple output fields")


class PromptStrategy(BaseModel):
    best_subset: List[Any] = []

    def validate_inputs(self, inputs_dict):
        if not set(inputs_dict.keys()) == set(self.input_variables.keys()):
            logger.error(f"Input keys do not match expected input keys Expected = {inputs_dict.keys()} Received = {self.input_variables.keys()}")
            raise ValueError(f"Input keys do not match expected input keys Expected: {inputs_dict.keys()} Received: {self.input_variables.keys()}")

    def format(self, **kwargs: Any) -> str:
        logger.debug(f"PromptStrategy format with kwargs: {kwargs}")
        return self.format_prompt(**kwargs)

    def format_prompt(self, **kwargs: Any) -> str:
        llm_type = kwargs.pop('llm_type', None)

        trained_state = kwargs.pop('trained_state', None)
        print_prompt = kwargs.pop('print_prompt', False)
        use_training = kwargs.pop('use_training', True)
        examples = kwargs.pop('__examples__', self.__examples__)  # Add this line

        # print(f"Formatting prompt with trained_state {trained_state} and print_prompt {print_prompt} and kwargs {kwargs}")
        # print(f"Formatting prompt with use_training {use_training}")

        try:
            # logger.debug(f"Formatting prompt with kwargs: {kwargs}")
            self.validate_inputs(kwargs)

            # logger.debug(f"PromptStrategy format_prompt with kwargs: {kwargs}")

            if llm_type == 'openai':
                prompt = self._format_openai_prompt(trained_state, use_training, examples, **kwargs)
            elif llm_type == 'openai_json':
                prompt = self._format_openai_json_prompt(trained_state, use_training, examples, **kwargs)
            elif llm_type == 'anthropic':
                prompt = self._format_anthropic_prompt(trained_state, use_training, examples, **kwargs)

            return prompt
        except Exception as e:
            logger.error(f"Failed to format prompt with kwargs: {kwargs}")
            import traceback
            traceback.print_exc()
            raise e

    def parse_output_to_fields(self, output: str, llm_type: str) -> dict:
        if llm_type == 'openai_json':
            return self._parse_openai_json_output_to_fields(output)
        elif llm_type == 'openai':
            return self._parse_openai_output_to_fields(output)
        elif llm_type == 'anthropic':
            return self._parse_anthropic_output_to_fields(output)
        elif llm_type == 'test':
            return self._parse_openai_output_to_fields(output)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

    @abstractmethod
    def _format_openai_prompt(self, trained_state, use_training, examples, **kwargs) -> str:
        pass

    @abstractmethod
    def _format_openai_json_prompt(self, trained_state, use_training, examples, **kwargs) -> str:
        pass

    @abstractmethod
    def _format_anthropic_prompt(self, trained_state, use_training, examples, **kwargs) -> str:
        pass

    def _get_output_field(self, field_name):
        for output_name, output_field in self.output_variables.items():
            if output_field.name == field_name:
                return output_name

    @abstractmethod
    def _parse_openai_output_to_fields(self, output: str) -> dict:
        pass

    @abstractmethod
    def _parse_anthropic_output_to_fields(self, output: str) -> dict:
        pass

    @abstractmethod
    def _parse_openai_json_output_to_fields(self, output: str) -> dict:
        pass


class DefaultPromptStrategy(PromptStrategy):
    OUTPUT_TOKEN = "🔑"

    def _format_openai_json_prompt(self, trained_state, use_training, examples, **kwargs) -> str:
        prompt = "Follow the following format. Answer with a JSON object. Attributes that have values should not be changed or repeated."

        if len(self.output_variables) > 1:
            output_field_names = ', '.join([output_field.name for output_field in self.output_variables.values()])
            prompt += f" Provide answers for {output_field_names}.\n"

        if self.hint_variables:
            prompt += "\n"

            for _, hint_field in self.hint_variables.items():
                prompt += hint_field.format_prompt_description("openai") + "\n"

        prompt += "\nInput Fields:\n"
        input_fields_dict = {}
        for input_name, input_field in self.input_variables.items():
            input_fields_dict[input_field.name] = input_field.desc
        prompt += json.dumps(input_fields_dict, indent=2) + "\n"

        prompt += "\nOutput Fields:\n"
        output_fields_dict = {}
        for output_name, output_field in self.output_variables.items():
            output_fields_dict[output_field.name] = output_field.desc
        prompt += json.dumps(output_fields_dict, indent=2) + "\n"

        if examples:
            prompt += "\nExamples:\n"
            for example_input, example_output in examples:
                example_dict = {"input": {}, "output": {}}
                for input_name, input_field in self.input_variables.items():
                    example_dict["input"].update(input_field.format_prompt_value_json(example_input.get(input_name), 'openai_json'))
                for output_name, output_field in self.output_variables.items():
                    if isinstance(example_output, dict):
                        example_dict["output"].update(output_field.format_prompt_value_json(example_output.get(output_name), 'openai_json'))
                    else:
                        example_dict["output"].update(output_field.format_prompt_value_json(example_output, 'openai_json'))
                prompt += json.dumps(example_dict, indent=2) + "\n"

        if trained_state and trained_state.examples and use_training:
            prompt += "\nTrained Examples:\n"
            for example_X, example_y in trained_state.examples:
                example_dict = {"input": {}, "output": {}}
                for input_name, input_field in self.input_variables.items():
                    example_dict["input"].update(input_field.format_prompt_value_json(example_X.get(input_name), 'openai_json'))
                for output_name, output_field in self.output_variables.items():
                    if isinstance(example_y, dict):
                        example_dict["output"].update(output_field.format_prompt_value_json(example_y.get(output_name), 'openai_json'))
                    else:
                        example_dict["output"].update(output_field.format_prompt_value_json(example_y, 'openai_json'))
                prompt += json.dumps(example_dict, indent=2) + "\n"

        prompt += "\nInput:\n"
        input_dict = {}
        for input_name, input_field in self.input_variables.items():
            input_dict.update(input_field.format_prompt_value_json(kwargs.get(input_name), 'openai_json'))
        prompt += json.dumps(input_dict, indent=2) + "\n"

        prompt += "\nOutput:\n"
        output_dict = {}
        for output_name, output_field in self.output_variables.items():
            output_dict.update(output_field.format_prompt_json('openai_json'))
        prompt += json.dumps(output_dict, indent=2) + "\n"

        return prompt

    def _format_openai_prompt(self, trained_state, use_training, examples, **kwargs) -> str:
        # print(f"Formatting prompt {kwargs}")
        prompt = "Follow the following format. Attributes that have values should not be changed or repeated. "

        if len(self.output_variables) > 1:
            #Provide answers for Solution Effectiveness, Rationale and Confidence
            # Extract names from output_variables
            output_field_names = ', '.join([output_field.name for output_field in self.output_variables.values()])

            # Format the instruction with the extracted names
            prompt += f"Provide answers for {output_field_names}\n"


        if self.hint_variables:
            prompt += "\n"

            for _, hint_field in self.hint_variables.items():
                prompt += hint_field.format_prompt_description("openai") + "\n"

        prompt += "\n\n"

        for input_name, input_field in self.input_variables.items():
            # prompt += f"⏎{input_field.name}: {input_field.desc}\n"
            prompt += input_field.format_prompt_description("openai") + "\n"

        for output_name, output_field in self.output_variables.items():
            prompt += output_field.format_prompt_description("openai") + "\n"
            # prompt += f"{self.OUTPUT_TOKEN}{output_field.name}: {output_field.desc}\n"

        if examples:
            for example_input, example_output in examples:
                prompt += "\n---\n\n"
                for input_name, input_field in self.input_variables.items():
                    prompt += input_field.format_prompt_value(example_input.get(input_name), "openai") + "\n"
                for output_name, output_field in self.output_variables.items():
                    if isinstance(example_output, dict):
                        prompt += output_field.format_prompt_value(example_output.get(output_name), "openai") + "\n"
                    else:
                        prompt += output_field.format_prompt_value(example_output, "openai") + "\n"

        if trained_state and trained_state.examples and use_training:
            for example_X, example_y in trained_state.examples:
                prompt += "\n---\n\n"

                for input_name, input_field in self.input_variables.items():
                    prompt += input_field.format_prompt_value(example_X.get(input_name), "openai") + "\n"

                for output_name, output_field in self.output_variables.items():
                    if isinstance(example_y, dict):
                        prompt += output_field.format_prompt_value(example_y.get(output_name), "openai") + "\n"
                    else:
                        prompt += output_field.format_prompt_value(example_y, "openai") + "\n"

        prompt += "\n---\n\n"


        for input_name, input_field in self.input_variables.items():
            prompt += input_field.format_prompt_value(kwargs.get(input_name), "openai") + "\n"

        for output_name, output_field in self.output_variables.items():
            prompt += output_field.format_prompt("openai") + "\n"

        return prompt

    def _format_anthropic_prompt(self, trained_state, use_training, examples, **kwargs) -> str:
        # print(f"Formatting prompt {kwargs}")
        # prompt = "Follow the following format. Attributes that have values should not be changed or repeated. "
        prompt = ""

        output_field_names = ', '.join([output_field.name for output_field in self.output_variables.values()])
        # Format the instruction with the extracted names
        prompt += f"Provide answers for output fields {output_field_names}. Follow the XML output format, only show the output fields do not repeat the hints, input fields or examples.\n"

        if self.hint_variables:
            prompt += "\n<hints>\n"
            for _, hint_field in self.hint_variables.items():
                prompt += hint_field.format_prompt_description("anthropic") + "\n"
            prompt += "</hints>\n"

        prompt += "\n\n<input_fields>\n"
        for input_name, input_field in self.input_variables.items():
            # prompt += f"⏎{input_field.name}: {input_field.desc}\n"
            prompt += input_field.format_prompt_description("anthropic") + "\n"
        prompt += "</input_fields>\n"
        prompt += "\n<output_fields>\n"
        for output_name, output_field in self.output_variables.items():
            prompt += output_field.format_prompt_description("anthropic") + "\n"
            # prompt += f"{self.OUTPUT_TOKEN}{output_field.name}: {output_field.desc}\n"
        prompt += "</output_fields>\n"

        if examples:
            prompt += "\n<examples>\n"
            for example_input, example_output in examples:
                prompt += "\n<example>\n"
                prompt += "<input>\n"
                for input_name, input_field in self.input_variables.items():
                    prompt += input_field.format_prompt_value(example_input.get(input_name), "anthropic") + "\n"
                prompt += "</input>\n"
                prompt += "<output>\n"
                for output_name, output_field in self.output_variables.items():
                    if isinstance(example_output, dict):
                        prompt += output_field.format_prompt_value(example_output.get(output_name), "anthropic") + "\n"
                    else:
                        prompt += output_field.format_prompt_value(example_output, "anthropic") + "\n"
                prompt += "</output>\n"
                prompt += "</example>\n"
            prompt += "</examples>\n"

        if trained_state and trained_state.examples and use_training:
            prompt += "\n<examples>\n"
            for example_X, example_y in trained_state.examples:
                prompt += "\n<example>\n"
                prompt += "<input>\n"
                for input_name, input_field in self.input_variables.items():
                    prompt += input_field.format_prompt_value(example_X.get(input_name), "anthropic") + "\n"
                prompt += "</input>\n"
                prompt += "<output>\n"
                for output_name, output_field in self.output_variables.items():
                    if isinstance(example_y, dict):
                        prompt += output_field.format_prompt_value(example_y.get(output_name), "anthropic") + "\n"
                    else:
                        prompt += output_field.format_prompt_value(example_y, "anthropic") + "\n"
                prompt += "</output>\n"
                prompt += "</example>\n"
            prompt += "</examples>\n"

        prompt += "\n<input>\n"
        for input_name, input_field in self.input_variables.items():
            prompt += input_field.format_prompt_value(kwargs.get(input_name), "anthropic") + "\n"
        prompt += "</input>\n"

        prompt += "\n<output>\n"
        for output_name, output_field in self.output_variables.items():
            prompt += output_field.format_prompt("anthropic") + "\n"
        prompt += "</output>\n"
        return prompt

    def _parse_openai_output_to_fields(self, output: str) -> dict:
        try:
            pattern = r'^([^:]+): (.*)'
            lines = output.split(self.OUTPUT_TOKEN)
            parsed_fields = {}
            logger.debug(f"Parsing output to fields with pattern {pattern} and lines {lines}")
            for line in lines:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    field_name, field_content = match.groups()
                    logger.debug(f"Matched line {line} - field name {field_name} field content {field_content}")
                    output_field = self._get_output_field(field_name)
                    if output_field:
                        logger.debug(f"Matched field {field_name} to output field {output_field}")
                        parsed_fields[output_field] = field_content
                    else:
                        logger.error(f"Field {field_name} not found in output variables")
                else:
                    logger.debug(f"NO MATCH line {line}")

            if len(self.output_variables) == 1:
                first_value = next(iter(parsed_fields.values()), None)
                if not first_value:
                    # logger.debug(f"NO MATCHES - setting last field to output: {lines[-1]}")
                    parsed_fields[list(self.output_variables.keys())[0]] = lines[-1]
                # else:
                #     logger.error(f"NO MATCHES - setting last field to output: {lines[-1]}")
            logger.debug(f"Parsed fields: {parsed_fields}")
            return parsed_fields
        except Exception as e:
            import traceback
            traceback.print_exc()

            raise e

    def _parse_anthropic_output_to_fields(self, output: str) -> dict:
        try:
            parsed_fields = {}
            for output_name, output_field in self.output_variables.items():
                pattern = fr"<{output_field.name}>(.*?)</{output_field.name}>"
                # match = re.search(pattern, output, re.DOTALL)
                # if match:
                #     parsed_fields[output_name] = match.group(1).strip()
                matches = re.findall(pattern, output, re.DOTALL)
                if matches:
                    # Take the last match
                    last_match = matches[-1]
                    parsed_fields[output_name] = last_match.strip()


            logger.debug(f"Parsed fields: {parsed_fields}")
            return parsed_fields
        except Exception as e:
            import traceback
            traceback.print_exc()

            raise e

    def _parse_openai_json_output_to_fields(self, output: str) -> dict:
        print(f"Parsing openai json")
        try:
            # Parse the JSON output
            json_output = json.loads(output)

            # Initialize an empty dictionary to store the parsed fields
            parsed_fields = {}

            # Iterate over the output variables
            for output_name, output_field in self.output_variables.items():
                # Check if the output field exists in the JSON output
                if output_field.name in json_output:
                    # Get the value of the output field from the JSON output
                    field_value = json_output[output_field.name]

                    # Apply any necessary transformations to the field value
                    transformed_value = output_field.transform_value(field_value)

                    # Store the transformed value in the parsed fields dictionary
                    parsed_fields[output_name] = transformed_value
                else:
                    # If the output field is not present in the JSON output, set its value to None
                    parsed_fields[output_name] = None

            logger.debug(f"Parsed fields: {parsed_fields}")
            return parsed_fields
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output: {e}")
            raise e
        except Exception as e:
            logger.error(f"An error occurred while parsing JSON output: {e}")
            raise e