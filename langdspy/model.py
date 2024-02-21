from langchain.prompts import BasePromptTemplate  # Assuming this is the correct import path
import time
import random
import re
from langchain.prompts import FewShotPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator, Extra, PrivateAttr
from langchain_core.pydantic_v1 import validator
from langchain_core.language_models import BaseLLM
from typing import Any, Dict, List, Type, Optional, Callable
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
import threading



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
    # prompt_history: List[str] = [] - Was trying to find a way to make a list of prompts for inspection 

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

            print(res)
            logger.debug(f"Raw output for prompt runner {self.template.__class__.__name__}: {res}")

            # Use the parse_output_to_fields method from the PromptStrategy
            parsed_output = {}
            try:
                parsed_output = self.template.parse_output_to_fields(res)
            except Exception as e:
                import traceback
                traceback.print_exc()
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

                    # Validate the transformed value
                    if not output_field.validate_value(input, output_value):
                        validation = False
                        logger.error(f"Failed to validate field {attr_name} value {output_value} for prompt runner {self.template.__class__.__name__}")

                    # Get the transformed value
                    try:
                        transformed_val = output_field.transform_value(output_value)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        logger.error(f"Failed to transform field {attr_name} value {output_value} for prompt runner {self.template.__class__.__name__}")
                        validation = False
                        continue

                    # Update the output with the transformed value
                    parsed_output[attr_name] = transformed_val

            res = {attr_name: parsed_output.get(attr_name, None) for attr_name in self.template.output_variables.keys()}

            if validation:
                # Return a dictionary keyed by attribute names with validated values
                return res

            logger.error(f"Output validation failed for prompt runner {self.template.__class__.__name__}, pausing before we retry")
            time.sleep(random.uniform(0.1, 1.5))
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


_multi_lock = threading.Lock()
class MultiPromptRunner(PromptRunner):
    predictions: List[Any] = []

    def __init__(self, template_class, prompt_strategy):
        super().__init__(template_class, prompt_strategy)
        self.predictions = []

    def invoke(self, input: Input, config: Optional[RunnableConfig] = {}) -> List[Output]:
        # logger.debug(f"MultiPromptRunner invoke with input {input} and config {config}")
        number_of_threads = config.get('number_of_threads', 1)
        target_runs = config.get('target_runs', 1)

        # logger.debug(f"MultiPromptRunner number_of_threads: {number_of_threads} target_runs: {target_runs}")
        predictions = []
        futures = []

        def run_task():
            with _multi_lock:
                logger.debug(f"Running task")
                if len(self.predictions) < target_runs:
                    # logger.debug(f"Running task with input {input} and config {config}")
                    prediction = super(MultiPromptRunner, self).invoke(input, config)
                    # logger.debug(f"Prediction: {prediction}")
                    self.predictions.append(prediction)

        with ThreadPoolExecutor(max_workers=number_of_threads) as executor:
            for _ in range(target_runs):
                # logger.debug(f"Submitting task to executor")
                future = executor.submit(run_task)
                futures.append(future)
                # logger.debug(f"Task submitted to executor")

            for future in as_completed(futures):
                future.result()  # This will block until the future is done


        # logger.debug(f"MultiPromptRunner predictions: {self.predictions}")
        return self.predictions




class Model(RunnableSerializable):
    prompt_runners = []

    def __init__(self):
        super().__init__()

        for field_name, field in self.__fields__.items():
            if issubclass(field.type_, PromptRunner):
                self.prompt_runners.append((field_name, field.default))
