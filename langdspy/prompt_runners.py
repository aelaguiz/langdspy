import time
import random
from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator, Extra, PrivateAttr
from langchain_core.pydantic_v1 import validator
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
#  from langchain_contrib.llms.testing import FakeLLM
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
prompt_logger = logging.getLogger("langdspy.prompts")

class Prediction(BaseModel):
    class Config:
        extra = Extra.allow  # This allows the model to accept extra fields that are not explicitly declared

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Initialize BaseModel with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)  # Dynamically assign attributes

class PromptHistory(BaseModel):
    history: List[Dict[str, Any]] = Field(default_factory=list)
    
    def add_entry(self, llm, prompt, llm_response, parsed_output, error, start_time, end_time):
        self.history.append({
            "duration_ms": round((end_time - start_time) * 1000),
            "llm": llm,
            "llm_response": llm_response,
            "parsed_output": parsed_output,
            "prompt": prompt,
            "error": error,
            "timestamp": end_time,
        })

    def reset(self):
        self.history = []


        
class PromptRunner(RunnableSerializable):
    template: PromptSignature = None
    # prompt_history: List[str] = [] - Was trying to find a way to make a list of prompts for inspection 
    model_kwargs: Dict[str, Any] = {}
    kwargs: Dict[str, Any] = {}
    prompt_history: PromptHistory = Field(default_factory=PromptHistory)


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

    def _determine_llm_type(self, llm):
        if isinstance(llm, ChatOpenAI):  # Assuming OpenAILLM is the class for OpenAI models
            logger.debug(llm.kwargs)
            if llm.kwargs.get('response_format', {}).get('type') == 'json_object':
                logger.info("OpenAI model response format is json_object")
                return 'openai_json'
            return 'openai'
        elif isinstance(llm, ChatAnthropic):  # Assuming AnthropicLLM is the class for Anthropic models
            return 'anthropic'
        else:
            return 'openai'  # Default to OpenAI if model type cannot be determined

    def _determine_llm_model(self, llm):
        if isinstance(llm, ChatOpenAI):  # Assuming OpenAILLM is the class for OpenAI models
            return llm.model_name
        elif isinstance(llm, ChatAnthropic):  # Assuming AnthropicLLM is the class for Anthropic models
            return llm.model
        elif hasattr(llm, 'model_name'):
            return llm.model_name
        elif hasattr(llm, 'model'):
            return llm.model
        else:
            return '???'

    def get_prompt_history(self):
        return self.prompt_history.history

    def clear_prompt_history(self):
        self.prompt_history.reset()
    
    def _invoke_with_retries(self, chain, input, max_tries=1, config: Optional[RunnableConfig] = {}):
        total_max_tries = max_tries
        hard_fail = config.get('hard_fail', True)
        llm_type, llm_model = self._get_llm_info(config)
        
        logger.debug(f"LLM type: {llm_type} - model {llm_model}")

        while max_tries >= 1:
            start_time = time.time()
            try:
                formatted_prompt, prompt_res = self._execute_prompt(chain, input, config, llm_type)
                parsed_output, validation_err = self._process_output(prompt_res, input, llm_type)
                
                end_time = time.time()
                self._log_prompt_history(config, formatted_prompt, prompt_res, parsed_output, validation_err, start_time, end_time)

                if validation_err is None:
                    return parsed_output

            except Exception as e:
                self._handle_exception(e, max_tries)

            max_tries -= 1
            if max_tries >= 1:
                self._handle_retry(max_tries)

        return self._handle_failure(hard_fail, total_max_tries, prompt_res)

    def _get_llm_info(self, config):
        llm_type = config.get('llm_type') or self._determine_llm_type(config['llm'])
        llm_model = self._determine_llm_model(config['llm'])
        return llm_type, llm_model

    def _execute_prompt(self, chain, input, config, llm_type):
        kwargs = {**self.model_kwargs, **self.kwargs}
        trained_state = self._get_trained_state(config)
        print_prompt = kwargs.get('print_prompt', config.get('print_prompt', False))
        
        invoke_args = {**input, 'print_prompt': print_prompt, **kwargs, 'trained_state': trained_state, 'use_training': config.get('use_training', True), 'llm_type': llm_type}
        formatted_prompt = self.template.format_prompt(**invoke_args)
        
        self._log_prompt(formatted_prompt, print_prompt)
        
        prompt_res = chain.invoke(invoke_args, config=config)
        return formatted_prompt, prompt_res

    def _get_trained_state(self, config):
        trained_state = config.get('trained_state') or self.model_kwargs.get('trained_state') or self.kwargs.get('trained_state')
        return trained_state if trained_state and trained_state.examples else None

    def _log_prompt(self, formatted_prompt, print_prompt):
        prompt_logger.info(f"------------------------PROMPT START--------------------------------")
        prompt_logger.info(formatted_prompt)
        prompt_logger.info(f"------------------------PROMPT END----------------------------------\n")

    def _process_output(self, prompt_res, input, llm_type):
        self._log_result(prompt_res)
        
        parsed_output = {}
        validation_err = None
        try:
            parsed_output = self.template.parse_output_to_fields(prompt_res, llm_type)
            validation_err = self._validate_output(parsed_output, input)
        except Exception as e:
            validation_err = f"Failed to parse output for prompt runner {self.template.__class__.__name__}"
            logger.error(validation_err)
            import traceback
            traceback.print_exc()

        return parsed_output, validation_err

    def _log_result(self, prompt_res):
        prompt_logger.info(f"------------------------RESULT START--------------------------------")
        prompt_logger.info(prompt_res)
        prompt_logger.info(f"------------------------RESULT END----------------------------------\n")

    def _validate_output(self, parsed_output, input):
        if len(parsed_output.keys()) != len(self.template.output_variables.keys()):
            return f"Output keys do not match expected output keys for prompt runner {self.template.__class__.__name__}"

        for attr_name, output_field in self.template.output_variables.items():
            output_value = parsed_output.get(attr_name)
            if not output_value:
                return f"Failed to get output value for field {attr_name} for prompt runner {self.template.__class__.__name__}"

            if not output_field.validate_value(input, output_value):
                return f"Failed to validate field {attr_name} value {output_value} for prompt runner {self.template.__class__.__name__}"

            try:
                parsed_output[attr_name] = output_field.transform_value(output_value)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"Failed to transform field {attr_name} value {output_value} for prompt runner {self.template.__class__.__name__}"

        return None

    def _log_prompt_history(self, config, formatted_prompt, prompt_res, parsed_output, validation_err, start_time, end_time):
        llm_info = f"{self._determine_llm_type(config['llm'])} {self._determine_llm_model(config['llm'])}"
        self.prompt_history.add_entry(llm_info, formatted_prompt, prompt_res, parsed_output, validation_err, start_time, end_time)

    def _handle_exception(self, e, max_tries):
        import traceback
        traceback.print_exc()
        logger.error(f"Failed in the LLM layer {e} - sleeping then trying again")
        time.sleep(random.uniform(0.1, 1.5))

    def _handle_retry(self, max_tries):
        logger.error(f"Output validation failed for prompt runner {self.template.__class__.__name__}, pausing before we retry")
        time.sleep(random.uniform(0.05, 0.25))

    def _handle_failure(self, hard_fail, total_max_tries, prompt_res):
        if hard_fail:
            raise ValueError(f"Output validation failed for prompt runner {self.template.__class__.__name__} after {total_max_tries} tries.")
        else:
            logger.error(f"Output validation failed for prompt runner {self.template.__class__.__name__} after {total_max_tries} tries, returning None.")
            if len(self.template.output_variables.keys()) == 1:
                return {attr_name: prompt_res for attr_name in self.template.output_variables.keys()}
            else:
                return {attr_name: None for attr_name in self.template.output_variables.keys()}
 
    def invoke(self, input: Input, config: Optional[RunnableConfig] = {}) -> Output:
        chain = (
            self.template
            | config['llm']
            | StrOutputParser()
        )

        max_retries = config.get('max_tries', 3)

        if '__examples__' in config:
            input['__examples__'] = config['__examples__']

        res = self._invoke_with_retries(chain, input, max_retries, config=config)

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
