import pytest
from enum import Enum
from langdspy.field_descriptors import InputField, InputFieldList, OutputField, OutputFieldEnum, OutputFieldEnumList, OutputFieldBool, OutputFieldChooseOne, InputFieldDict, InputFieldDictList

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

def test_output_field_choose_one_initialization():
    choices = ["Option 1", "Option 2", "Option 3"]
    field = OutputFieldChooseOne("name", "description", choices)
    assert field.name == "name"
    assert field.desc == "description"
    assert field.validator.__name__ == "is_one_of"
    assert field.kwargs['choices'] == choices

def test_output_field_choose_one_format_prompt_description():
    choices = ["Option 1", "Option 2", "Option 3"]

    field = OutputFieldChooseOne("name", "description", choices)
    assert "One of: Option 1, Option 2, Option 3" in field.format_prompt_description("openai")
    assert "One of: <choices>Option 1, Option 2, Option 3</choices>" in field.format_prompt_description("anthropic")

def test_format_prompt_value_openai():
    field = InputFieldDict("input_dict", "A dictionary input")
    input_dict = {
        "name": "John Doe",
        "age": 30,
        "city": "New York"
    }
    expected_output = "✅input_dict name: John Doe\n✅input_dict age: 30\n✅input_dict city: New York"
    print(field.format_prompt_value(input_dict, "openai"))
    print(expected_output)
    assert field.format_prompt_value(input_dict, "openai") == expected_output

def test_format_prompt_value_anthropic():
    field = InputFieldDict("input_dict", "A dictionary input")
    input_dict = {
        "name": "John Doe",
        "age": 30,
        "city": "New York"
    }
    expected_output = "<input_dict>\n  <name>John Doe</name>\n  <age>30</age>\n  <city>New York</city>\n</input_dict>"
    assert field.format_prompt_value(input_dict, "anthropic") == expected_output

# tests/test_input_field_dict_list.py
import pytest
from langdspy.field_descriptors import InputFieldDictList

def test_format_prompt_value_openai():
    field = InputFieldDictList("input_list", "A list of dictionaries")
    input_list = [
        {
            "name": "John Doe",
            "age": 30,
            "city": "New York"
        },
        {
            "name": "Jane Smith",
            "age": 25,
            "city": "London"
        }
    ]
    expected_output = "✅input_list Item 1:\n  name: John Doe\n  age: 30\n  city: New York\n\n✅input_list Item 2:\n  name: Jane Smith\n  age: 25\n  city: London"
    print(field.format_prompt_value(input_list, "openai"))
    print(expected_output)
    assert field.format_prompt_value(input_list, "openai") == expected_output

def test_format_prompt_value_anthropic():
    field = InputFieldDictList("input_list", "A list of dictionaries")
    input_list = [
        {
            "name": "John Doe",
            "age": 30,
            "city": "New York"
        },
        {
            "name": "Jane Smith",
            "age": 25,
            "city": "London"
        }
    ]
    expected_output = "<input_list>\n  <input_list 1>\n    <name>John Doe</name>\n    <age>30</age>\n    <city>New York</city>\n  </input_list 1>\n  <input_list 2>\n    <name>Jane Smith</name>\n    <age>25</age>\n    <city>London</city>\n  </input_list 2>\n</input_list>"
    assert field.format_prompt_value(input_list, "anthropic") == expected_output