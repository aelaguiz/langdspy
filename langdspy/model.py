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

from .field_descriptors import InputField, OutputField
from .prompt_strategies import PromptSignature, PromptStrategy

logger = logging.getLogger(__name__)

class Prediction(BaseModel):
    class Config:
        extra = Extra.allow  # This allows the model to accept extra fields that are not explicitly declared

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Initialize BaseModel with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)  # Dynamically assign attributes

        
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
            parsed_output = {}
            try:
                parsed_output = self.template.parse_output_to_fields(res)
            except:
                logger.error(f"Failed to parse output for prompt runner {self.template.__class__.__name__}")
                validation = False
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

            res = {attr_name: parsed_output.get(attr_name, None) for attr_name in self.template.output_variables.keys()}

            if validation:
                # Return a dictionary keyed by attribute names with validated values
                return res

            logger.error(f"Output validation failed for prompt runner {self.template.__class__.__name__}")
            max_tries -= 1

        if hard_fail:
            raise ValueError(f"Output validation failed for prompt runner {self.template.__class__.__name__} after {total_max_tries} tries.")
        else:
            logger.error(f"Output validation failed for prompt runner {self.template.__class__.__name__} after {total_max_tries} tries, returning unvalidated output.")

        return res
 
    def invoke(self, input: Input, config: Optional[RunnableConfig] = {}) -> Output:
        # logger.debug(f"Template: {self.template}")
        # logger.debug(f"Config: {config}")
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
