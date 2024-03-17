from typing import Any, Dict, List, Type, Optional, Callable
from langchain_core.runnables.utils import (
    Input,
    Output
)
from enum import Enum
from . import validators
from . import transformers

class FieldDescriptor:
    def __init__(self, name:str, desc: str, formatter: Optional[Callable[[Any], Any]] = None, transformer: Optional[Callable[[Any], Any]] = None, validator: Optional[Callable[[Any], Any]] = None, **kwargs):
        assert "⏎" not in name, "Field name cannot contain newline character"
        assert ":" not in name, "Field name cannot contain colon character"
        self.name = name
        self.desc = desc
        self.formatter = formatter
        self.transformer = transformer
        self.validator = validator
        self.kwargs = kwargs

    def format_value(self, value: Any) -> Any:
        if self.formatter:
            return self.formatter(value, self.kwargs)
        else:
            return value

    def transform_value(self, value: Any) -> Any:
        if self.transformer:
            return self.transformer(value, self.kwargs)
        else:
            return value

    def validate_value(self, input: Input, value: Any) -> bool:
        if self.validator:
            return self.validator(input, value, self.kwargs)
        else:
            return True

class HintField(FieldDescriptor):
    HINT_TOKEN_OPENAI = "💡"
    HINT_TOKEN_ANTHROPIC = None

    def __init__(self, desc: str, formatter: Optional[Callable[[Any], Any]] = None, transformer: Optional[Callable[[Any], Any]] = None, validator: Optional[Callable[[Any], Any]] = None, **kwargs):
        # Provide a default value for the name parameter, such as an empty string
        super().__init__("", desc, formatter, transformer, validator, **kwargs)

    def _start_format_openai(self):
        return f"{self.HINT_TOKEN_OPENAI}"

    def _start_format_anthropic(self):
        return f"<hint>"

    def format_prompt_description(self, llm_type: str):
        if llm_type == "openai":
            return f"{self._start_format_openai()} {self.desc}"
        elif llm_type == "anthropic":
            return f"{self._start_format_anthropic()}{self.desc}</hint>"

class InputField(FieldDescriptor):
    START_TOKEN_OPENAI = "✅"
    START_TOKEN_ANTHROPIC = None

    def _start_format_openai(self):
        return f"{self.START_TOKEN_OPENAI}{self.name}"

    def _start_format_anthropic(self):
        return f"<{self.name}>"

    def format_prompt_description(self, llm_type: str):
        if llm_type == "openai":
            return f"{self._start_format_openai()}: {self.desc}"
        elif llm_type == "anthropic":
            return f"{self._start_format_anthropic()}: {self.desc}"

    def format_prompt_value(self, value, llm_type: str):
        value = self.format_value(value)
        if llm_type == "openai":
            return f"{self._start_format_openai()}: {value}"
        elif llm_type == "anthropic":
            return f"{self._start_format_anthropic()}{value}</{self.name}>"

class InputFieldList(InputField):
    def format_prompt_description(self, llm_type: str):
        if llm_type == "openai":
            return f"{self._start_format_openai()}: {self.desc}"
        elif llm_type == "anthropic":
            return f"{self._start_format_anthropic()}: {self.desc}"

    def format_prompt_value(self, value, llm_type: str):
        res = ""
        if len(value) >= 1:
            for i, value in enumerate(value):
                if i > 0:
                    res += "\n"
                value = self.format_value(value)
                if llm_type == "openai":
                    res += f"{self.START_TOKEN_OPENAI} [{i}]: {value}"
                elif llm_type == "anthropic":
                    res += f"<item>{value}</item>"
        else:
            if llm_type == "openai":
                res += f"{self._start_format_openai()}: NO VALUES SPECIFIED"
            elif llm_type == "anthropic":
                res += f"{self._start_format_anthropic()}NO VALUES SPECIFIED</{self.name}>"

        return res

class OutputField(FieldDescriptor):
    START_TOKEN_OPENAI = "🔑"
    START_TOKEN_ANTHROPIC = None

    def _start_format_openai(self):
        return f"{self.START_TOKEN_OPENAI}{self.name}"

    def _start_format_anthropic(self):
        return f"<{self.name}>"

    def format_prompt_description(self, llm_type: str):
        if llm_type == "openai":
            return f"{self._start_format_openai()}: {self.desc}"
        elif llm_type == "anthropic":
            return f"{self._start_format_anthropic()}: {self.desc}"

    def format_prompt_value(self, value, llm_type: str):
        value = self.format_value(value)
        if llm_type == "openai":
            return f"{self._start_format_openai()}: {value}"
        elif llm_type == "anthropic":
            return f"{self._start_format_anthropic()}{value}</{self.name}>"

    def format_prompt(self, llm_type: str):
        if llm_type == "openai":
            return f"{self._start_format_openai()}:"
        elif llm_type == "anthropic":
            return f"{self._start_format_anthropic()}</{self.name}>"

class OutputFieldEnum(OutputField):
    def __init__(self, name: str, desc: str, enum: Enum, **kwargs):
        kwargs['enum'] = enum
        if not 'transformer' in kwargs:
            kwargs['transformer'] = transformers.as_enum
        if not 'validator' in kwargs:
            kwargs['validator'] = validators.is_one_of
            kwargs['choices'] = [e.name for e in enum]
        super().__init__(name, desc, **kwargs)

    def format_prompt_description(self, llm_type: str):
        enum = self.kwargs.get('enum')
        choices_str = ", ".join([e.name for e in enum])
        if llm_type == "openai":
            return f"{self._start_format_openai()}: One of: {choices_str} - {self.desc}"
        elif llm_type == "anthropic":
            return f"{self._start_format_anthropic()}: One of: {choices_str} - {self.desc}"