import time
import random
from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator, Extra, PrivateAttr
from langchain_core.pydantic_v1 import validator
from typing import Any, Dict, List, Type, Optional, Callable
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional



from langchain_core.runnables.utils import (
    Input,
    Output
)
from langchain_core.runnables.config import (
    RunnableConfig
)

import logging

from .prompt_strategies import PromptSignature

logger = logging.getLogger("langdspy")

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
    model_kwargs: Dict[str, Any] = {}
    kwargs: Dict[str, Any] = {}

    def __init__(self, template_class, prompt_strategy, **kwargs):
        super().__init__()

        self.kwargs = kwargs

        cls_ = type(template_class.__name__, (prompt_strategy, template_class), {})
        self.template = cls_()
    
    @validator("template")
    def check_template(
        cls, value: PromptSignature
    ) -> PromptSignature:
        return value

        
    def set_model_kwargs(self, model_kwargs):
        self.model_kwargs.update(model_kwargs)
    
    def _invoke_with_retries(self, chain, input, max_tries=1, config: Optional[RunnableConfig] = {}):
        total_max_tries = max_tries

        hard_fail = config.get('hard_fail', False)

        res = {}

        while max_tries >= 1:
            try:
                kwargs = {**self.model_kwargs, **self.kwargs}
                res = chain.invoke({**input, 'trained_state': config.get('trained_state', None), 'print_prompt': config.get('print_prompt', False), **kwargs}, config=config)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Failed in the LLM layer {e} - sleeping then trying again")
                time.sleep(random.uniform(0.1, 1.5))
                max_tries -= 1
                continue

            validation = True

            # logger.debug(f"Raw output for prompt runner {self.template.__class__.__name__}: {res}")

            # Use the parse_output_to_fields method from the PromptStrategy
            parsed_output = {}
            try:
                parsed_output = self.template.parse_output_to_fields(res)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Failed to parse output for prompt runner {self.template.__class__.__name__}")
                validation = False
            # logger.debug(f"Parsed output: {parsed_output}")

            len_parsed_output = len(parsed_output.keys())
            len_output_variables = len(self.template.output_variables.keys())
            # logger.debug(f"Parsed output keys: {parsed_output.keys()} [{len_parsed_output}] Expected output keys: {self.template.output_variables.keys()} [{len_output_variables}]")

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

        if '__examples__' in config:
            input['__examples__'] = config['__examples__']

        res = self._invoke_with_retries(chain, input, max_retries, config=config)

        # logger.debug(f"Result: {res}")

        prediction_data = {**input, **res}


        prediction = Prediction(**prediction_data)

        return prediction


class MultiPromptRunner(PromptRunner):
    def __init__(self, template_class, prompt_strategy):
        super().__init__(template_class, prompt_strategy)

    def invoke(self, input: Input, config: Optional[RunnableConfig] = {}) -> List[Output]:
        # logger.debug(f"MultiPromptRunner invoke with input {input} and config {config}")
        number_of_threads = config.get('number_of_threads', 1)
        target_runs = config.get('target_runs', 1)

        # logger.debug(f"MultiPromptRunner number_of_threads: {number_of_threads} target_runs: {target_runs}")
        predictions = []
        futures = []

        def run_task():
            # Direct invocation of the super class method without modifying shared state here
            return super(MultiPromptRunner, self).invoke(input, config)

        with ThreadPoolExecutor(max_workers=number_of_threads) as executor:
            for _ in range(target_runs):
                future = executor.submit(run_task)
                futures.append(future)

            for future in as_completed(futures):
                # Collect results as they complete
                prediction = future.result()
                predictions.append(prediction)


        # logger.debug(f"MultiPromptRunner predictions: {self.predictions}")
        return predictions