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
            return f"{self._start_format_anthropic()}{self.desc}</{self.name}>"
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
            return f"{self._start_format_anthropic()}{self.desc}</{self.name}>"
    def format_prompt_value(self, value, llm_type: str):
        res = ""
        if len(value) >= 1:
            if llm_type == "anthropic":
                res += f"<{self.name}>\n"
            for i, value in enumerate(value):
                if i > 0:
                    res += "\n"
                value = self.format_value(value)
                if llm_type == "openai":
                    res += f"{self.START_TOKEN_OPENAI} [{i}]: {value}"
                elif llm_type == "anthropic":
                    res += f"<item>{value}</item>"
            if llm_type == "anthropic":
                res += f"\n</{self.name}>"
        else:
            if llm_type == "openai":
                res += f"{self._start_format_openai()}: NO VALUES SPECIFIED"
            elif llm_type == "anthropic":
                res += f"{self._start_format_anthropic()}NO VALUES SPECIFIED</{self.name}>"
        return res

class InputFieldDict(InputField):
    def format_prompt_value(self, value, llm_type: str):
        if llm_type == "openai":
            return self._format_openai_prompt_value(value)
        elif llm_type == "anthropic":
            return self._format_anthropic_prompt_value(value)

    def _format_openai_prompt_value(self, value):
        formatted_dict = ""
        for key, val in value.items():
            formatted_dict += f"{self._start_format_openai()} {key}: {val}\n"
        return formatted_dict.strip()

    def _format_anthropic_prompt_value(self, value):
        formatted_dict = f"<{self.name}>\n"
        for key, val in value.items():
            formatted_dict += f"  <{key}>{val}</{key}>\n"
        formatted_dict += f"</{self.name}>"
        return formatted_dict

class InputFieldDictList(InputField):
    def format_prompt_value(self, value, llm_type: str):
        if llm_type == "openai":
            return self._format_openai_prompt_value(value)
        elif llm_type == "anthropic":
            return self._format_anthropic_prompt_value(value)

    def _format_openai_prompt_value(self, value):
        formatted_list = ""
        for i, item in enumerate(value):
            formatted_list += f"{self._start_format_openai()} Item {i+1}:\n"
            for key, val in item.items():
                formatted_list += f"  {key}: {val}\n"
            formatted_list += "\n"
        return formatted_list.strip()

    def _format_anthropic_prompt_value(self, value):
        formatted_list = f"<{self.name}>\n"
        for i, item in enumerate(value):
            formatted_list += f"  <{self.name} {i + 1}>\n"
            for key, val in item.items():
                formatted_list += f"    <{key}>{val}</{key}>\n"
            formatted_list += f"  </{self.name} {i + 1}>\n"
        formatted_list += f"</{self.name}>"
        return formatted_list

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
            return f"{self._start_format_anthropic()}{self.desc}</{self.name}>"
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

class OutputFieldBool(OutputField):
    def __init__(self, name: str, desc: str, **kwargs):
        if not 'transformer' in kwargs:
            kwargs['transformer'] = transformers.as_bool
        if not 'validator' in kwargs:
            kwargs['validator'] = validators.is_one_of
            kwargs['choices'] = ['Yes', 'No']

        super().__init__(name, desc, **kwargs)

    def format_prompt_description(self, llm_type: str):
        choices_str = ", ".join(['Yes', 'No'])
        if llm_type == "openai":
            return f"{self._start_format_openai()}: One of: {choices_str} - {self.desc}"
        elif llm_type == "anthropic":
            return f"{self._start_format_anthropic()}One of: <choices>{choices_str}</choices> - {self.desc}</{self.name}>"

class OutputFieldChooseOne(OutputField):
    def __init__(self, name: str, desc: str, choices: List[str], **kwargs):
        kwargs['choices'] = choices

        if not 'validator' in kwargs:
            kwargs['validator'] = validators.is_one_of
            kwargs['choices'] = choices
        super().__init__(name, desc, **kwargs)
    
    def format_prompt_description(self, llm_type: str):
        choices_str = ", ".join(self.kwargs.get('choices', []))
        if llm_type == "openai":
            return f"{self._start_format_openai()}: One of: {choices_str} - {self.desc}"
        elif llm_type == "anthropic":
            return f"{self._start_format_anthropic()}One of: <choices>{choices_str}</choices> - {self.desc}</{self.name}>"
        
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
            return f"{self._start_format_anthropic()}One of: <choices>{choices_str}</choices> - {self.desc}</{self.name}>"

class OutputFieldEnumList(OutputField):
    def __init__(self, name: str, desc: str, enum: Enum, **kwargs):
        kwargs['enum'] = enum
        if not 'transformer' in kwargs:
            kwargs['transformer'] = transformers.as_enum_list
        if not 'validator' in kwargs:
            kwargs['validator'] = validators.is_subset_of
            kwargs['choices'] = [e.name for e in enum]
        super().__init__(name, desc, **kwargs)
    
    def format_prompt_description(self, llm_type: str):
        enum = self.kwargs.get('enum')
        choices_str = ", ".join([e.name for e in enum])
        if llm_type == "openai":
            return f"{self._start_format_openai()}: A comma-separated list of one or more of: {choices_str} - {self.desc}"
        elif llm_type == "anthropic":
            return f"{self._start_format_anthropic()}A comma-separated list of one or more of: <choices>{choices_str}</choices> - {self.desc}</{self.name}>"