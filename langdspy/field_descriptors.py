from typing import Any, Dict, List, Type, Optional, Callable
from langchain_core.runnables.utils import (
    Input,
    Output
)
import logging

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
    START_TOKEN = "⏎"

    def _start_format(self):
        return f"{self.START_TOKEN}{self.name}"
        
    def format_prompt_description(self):
        return f"{self._start_format()}: {self.desc}"

    def format_prompt_value(self, value):
        value = self.format_value(value)
        return f"{self._start_format()}: {value}"

class InputFieldList(InputField):
    def format_prompt_description(self):
        return f"{self._start_format()}: {self.desc}"

    def format_prompt_value(self, value):
        res = ""
        for i, value in enumerate(value):
            if i > 0:
                res += "\n"
            value = self.format_value(value)
            res += f"{self.START_TOKEN}{self.name} [{i}]: {value}"

        # import traceback
        # traceback.print_stack()
        # input(f"Hit enter but res: {res}...")
        return res

class OutputField(FieldDescriptor):
    pass
