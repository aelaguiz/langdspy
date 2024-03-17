import pytest
from langdspy.field_descriptors import InputField, InputFieldList

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