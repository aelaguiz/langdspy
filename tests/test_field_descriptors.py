import pytest
from enum import Enum
from langdspy.field_descriptors import InputField, InputFieldList, OutputField, OutputFieldEnum, OutputFieldEnumList, OutputFieldBool

def test_input_field_initialization():
    field = InputField("name", "description")
    assert field.name == "name"
    assert field.desc == "description"
    assert field.formatter is None
    assert field.transformer is None
    assert field.validator is None

def test_input_field_format_prompt_description():
    field = InputField("name", "description")
    assert field.format_prompt_description("openai") == "✅name: description"

def test_input_field_format_prompt_value():
    field = InputField("name", "description")
    assert field.format_prompt_value("value", "openai") == "✅name: value"

def test_input_field_list_initialization():
    field = InputFieldList("name", "description")
    assert field.name == "name"
    assert field.desc == "description"
    assert field.formatter is None
    assert field.transformer is None
    assert field.validator is None

def test_input_field_list_format_prompt_description():
    field = InputFieldList("name", "description")
    assert field.format_prompt_description("openai") == "✅name: description"

def test_input_field_list_format_prompt_value():
    field = InputFieldList("name", "description")
    assert field.format_prompt_value(["value1", "value2"], "openai") == "✅ [0]: value1\n✅ [1]: value2"

def test_input_field_list_format_prompt_value_empty():
    field = InputFieldList("name", "description")
    assert field.format_prompt_value([], "openai") == "✅name: NO VALUES SPECIFIED"

class TestEnum(Enum):
    VALUE1 = "value1"
    VALUE2 = "value2"
    VALUE3 = "value3"

def test_output_field_enum_list_initialization():
    field = OutputFieldEnumList("name", "description", TestEnum)
    assert field.name == "name"
    assert field.desc == "description"
    print(field.kwargs)
    assert field.kwargs['enum'] == TestEnum
    assert field.transformer.__name__ == "as_enum_list"
    # assert field.kwargs['transformer'].__name__ == "as_enum_list"
    assert field.validator.__name__ == "is_subset_of"
    # assert field.kwargs['validator'].__name__ == "is_subset_of"
    assert field.kwargs['choices'] == ["VALUE1", "VALUE2", "VALUE3"]

def test_output_field_enum_list_format_prompt_description():
    field = OutputFieldEnumList("name", "description", TestEnum)
    assert "A comma-separated list of one or more of: VALUE1, VALUE2, VALUE3" in field.format_prompt_description("openai")
    assert "A comma-separated list of one or more of: <choices>VALUE1, VALUE2, VALUE3</choices>" in field.format_prompt_description("anthropic")


def test_output_field_bool_initialization():
    field = OutputFieldBool("name", "description")
    assert field.name == "name"
    assert field.desc == "description"
    assert field.transformer.__name__ == "as_bool"
    assert field.validator.__name__ == "is_one_of"
    assert field.kwargs['choices'] == ["Yes", "No"]

def test_output_field_bool_format_prompt_description():
    field = OutputFieldBool("name", "description")
    assert "One of: Yes, No" in field.format_prompt_description("openai")
    assert "One of: <choices>Yes, No</choices>" in field.format_prompt_description("anthropic")