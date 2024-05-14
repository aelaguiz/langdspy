import pytest
from langdspy.field_descriptors import InputField, OutputField, InputFieldList
from langdspy.formatters import as_multiline, as_bulleted_list, as_yaml

def test_input_field_with_formatter_json():
    field = InputField("description", "The description", formatter=as_multiline)
    input_value = "This is a\nmulti-line description."
    expected_output = {"description": "«This is a\nmulti-line description.»"}
    assert field.format_prompt_value_json(input_value, 'openai_json') == expected_output

def test_output_field_with_formatter_json():
    field = OutputField("items", "The list of items", formatter=as_bulleted_list)
    output_value = ["Item 1", "Item 2", "Item 3"]
    expected_output = {"items": "\n- Item 1\n- Item 2\n- Item 3"}

    print(field.format_prompt_value_json(output_value, 'openai_json'))
    assert field.format_prompt_value_json(output_value, 'openai_json') == expected_output

def test_input_field_list_with_formatter_json():
    field = InputFieldList("items", "The list of items", formatter=as_yaml)
    input_value = [
        {"name": "Item 1", "price": 10.99},
        {"name": "Item 2", "price": 5.99},
        {"name": "Item 3", "price": 8.99}
    ]
    expected_output = {
        "items": ['\n«name: Item 1\nprice: 10.99\n»\n', '\n«name: Item 2\nprice: 5.99\n»\n', '\n«name: Item 3\nprice: 8.99\n»\n']
    }
    print(expected_output)
    print(field.format_prompt_value_json(input_value, 'openai_json'))
    assert field.format_prompt_value_json(input_value, 'openai_json') == expected_output