from langchain.prompts import BasePromptTemplate  # Assuming this is the correct import path
import json
import re
from langchain.prompts import FewShotPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator, Extra
from langchain_core.pydantic_v1 import validator
from langchain_core.language_models import BaseLLM
from typing import Any, Dict, List, Type, Optional, Callable, Tuple, Union
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

    def validate_inputs(self, inputs_dict):
        expected_keys = set(self.input_variables.keys())
        received_keys = set(inputs_dict.keys())
        
        if expected_keys != received_keys:
            missing_keys = expected_keys - received_keys
            unexpected_keys = received_keys - expected_keys
            error_message = []
            
            if missing_keys:
                error_message.append(f"Missing input keys: {', '.join(missing_keys)}")
                logger.error(f"Missing input keys: {missing_keys}")
            if unexpected_keys:
                error_message.append(f"Unexpected input keys: {', '.join(unexpected_keys)}")
                logger.error(f"Unexpected input keys: {unexpected_keys}")
            
            error_message.append(f"Expected keys: {', '.join(expected_keys)}")
            error_message.append(f"Received keys: {', '.join(received_keys)}")
            
            logger.error(f"Input keys do not match expected input keys. Expected: {expected_keys}, Received: {received_keys}")
            raise ValueError(". ".join(error_message))

    '''
    def _validate_input(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.input_variables:
            return input_dict  # Return the input as-is if there are no input variables defined

        validated_input = {}
        for name, field in self.input_variables.items():
            if name not in input_dict:
                if not field.kwargs.get('optional', False):
                    raise ValueError(f"Missing required input: {name}")
                else:
                    validated_input[name] = None
                    continue
            value = input_dict[name]
            if not field.validate_value({}, value):
                raise ValueError(f"Invalid input for {name}: {value}")
            validated_input[name] = field.transform_value(value)
        return validated_input
    '''


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
        expected_keys = set(self.input_variables.keys())
        received_keys = set(inputs_dict.keys())
        
        if expected_keys != received_keys:
            missing_keys = expected_keys - received_keys
            unexpected_keys = received_keys - expected_keys
            error_message = []
            
            if missing_keys:
                error_message.append(f"Missing input keys: {', '.join(missing_keys)}")
                logger.error(f"Missing input keys: {missing_keys}")
            if unexpected_keys:
                error_message.append(f"Unexpected input keys: {', '.join(unexpected_keys)}")
                logger.error(f"Unexpected input keys: {unexpected_keys}")
            
            error_message.append(f"Expected keys: {', '.join(expected_keys)}")
            error_message.append(f"Received keys: {', '.join(received_keys)}")
            
            logger.error(f"Input keys do not match expected input keys. Expected: {expected_keys}, Received: {received_keys}")
            raise ValueError(". ".join(error_message))

    def format(self, **kwargs: Any) -> str:
        logger.debug(f"PromptStrategy format with kwargs: {kwargs}")
        return self.format_prompt(**kwargs)

    def format_prompt(self, **kwargs: Any) -> str:
        llm_type = kwargs.pop('llm_type', None)

        trained_state = kwargs.pop('trained_state', None)
        use_training = kwargs.pop('use_training', True)
        examples = kwargs.pop('__examples__', self.__examples__)  # Add this line

        try:
            
            self.validate_inputs(kwargs)

            if llm_type == 'openai':
                prompt = self._format_openai_prompt(trained_state, use_training, examples, **kwargs)
            elif llm_type == 'openai_json':
                prompt = self._format_openai_json_prompt(trained_state, use_training, examples, **kwargs)
            elif llm_type == 'anthropic' or llm_type == 'fake_anthropic':
                prompt = self._format_anthropic_prompt(trained_state, use_training, examples, **kwargs)
            else:
                raise ValueError(f"Unsupported LLM type: {llm_type}")

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
        elif llm_type == 'anthropic' or llm_type == 'fake_anthropic':
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
    def _parse_openai_output_to_fields(self, output: Union[str, 'AIMessage']) -> dict:
        pass

    @abstractmethod
    def _parse_anthropic_output_to_fields(self, output: str) -> dict:
        pass

    @abstractmethod
    def _parse_openai_json_output_to_fields(self, output: Union[str, 'AIMessage']) -> dict:
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


    def _format_anthopic_cache(self, trained_state, use_training, examples) -> str:
        cache = ""
        
        cache = f"Provide answers for output fields {', '.join([output_field.name for output_field in self.output_variables.values()])}. Follow the XML output format, only show the output fields do not repeat the hints, input fields or examples."
        
        # Hints
        if self.hint_variables:
            hint_content = "\n".join([hint_field.format_prompt_description("anthropic") for _, hint_field in self.hint_variables.items()])
            cache += f"Hints:\n{hint_content}"

        # Input and Output fields description
        fields_description = "<input_fields>\n"
        fields_description += "\n".join([input_field.format_prompt_description("anthropic") for _, input_field in self.input_variables.items()])
        fields_description += "\n</input_fields>\n<output_fields>\n"
        fields_description += "\n".join([output_field.format_prompt_description("anthropic") for _, output_field in self.output_variables.items()])
        fields_description += "\n</output_fields>"
        cache += fields_description

        # Examples
        if examples:
            for example_input, example_output in examples:
                example_message = "<example>\n<input>\n"
                example_message += "\n".join([input_field.format_prompt_value(example_input.get(input_name), "anthropic") for input_name, input_field in self.input_variables.items()])
                example_message += "\n</input>\n<output>\n"
                if isinstance(example_output, dict):
                    example_message += "\n".join([output_field.format_prompt_value(example_output.get(output_name), "anthropic") for output_name, output_field in self.output_variables.items()])
                else:
                    example_message += "\n".join([output_field.format_prompt_value(example_output, "anthropic") for output_name, output_field in self.output_variables.items()])
                example_message += "\n</output>\n</example>"
                cache += example_message

        # Trained examples
        if trained_state and trained_state.examples and use_training:
            for example_X, example_y in trained_state.examples:
                trained_example_message = "<example>\n<input>\n"
                trained_example_message += "\n".join([input_field.format_prompt_value(example_X.get(input_name), "anthropic") for input_name, input_field in self.input_variables.items()])
                trained_example_message += "\n</input>\n<output>\n"
                if isinstance(example_y, dict):
                    trained_example_message += "\n".join([output_field.format_prompt_value(example_y.get(output_name), "anthropic") for output_name, output_field in self.output_variables.items()])
                else:
                    trained_example_message += "\n".join([output_field.format_prompt_value(example_y, "anthropic") for output_name, output_field in self.output_variables.items()])
                trained_example_message += "\n</output>\n</example>"
                cache += trained_example_message

        system_message = SystemMessage(content="""[
            {
                type: "text",
                text: "%s",
            
                // Tell Anthropic to cache this block
                cache_control: { type: "ephemeral" },
            },
            ]""" % cache
        )

        return system_message


    def _format_anthropic_prompt(self, trained_state, use_training, examples, **kwargs) -> str:
        messages = []

        system_message = self._format_anthopic_cache(trained_state, use_training, examples)
        messages.append(system_message)

        human_content = ""

        # User input
        user_input = "<input>\n"
        user_input += "\n".join([input_field.format_prompt_value(kwargs.get(input_name), "anthropic") for input_name, input_field in self.input_variables.items()])
        user_input += "\n</input>"
        human_content += user_input

        # Assistant response format
        human_content += "Respond with the output in the following format:\n<output>\n[Your response here]\n</output>"

        messages.append(HumanMessage(human_content))

        return messages

    def _parse_openai_output_to_fields(self, output: str) -> dict:
        try:
            if isinstance(output, dict):
                return output

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

    def _parse_anthropic_output_to_fields(self, output: Union[str, 'AIMessage']) -> dict:
        try:
            # Extract content if output is an AIMessage
            if hasattr(output, 'content'):
                output = output.content

            print(self)
            print(f"output: {output}")    

            print(f"output_variables: {self.output_variables}")
            
            parsed_fields = {}
            for output_name, output_field in self.output_variables.items():
                if isinstance(output, str):
                    pattern = fr"<{output_field.name}>(.*?)</{output_field.name}>"
                    matches = re.findall(pattern, output, re.DOTALL)
                    if matches:
                        # Take the last match
                        last_match = matches[-1]
                        parsed_fields[output_name] = last_match.strip()
                
                elif isinstance(output, dict):
                    if output_name in output.keys():
                        parsed_fields[output_name] = output[output_name]
                else:
                    raise ValueError(f"Invalid output type: {type(output)}")

            logger.debug(f"Parsed fields: {parsed_fields}")
            return parsed_fields
        except Exception as e:
            logger.error(f"Error parsing Anthropic output: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

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
