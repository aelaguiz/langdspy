import pytest
from langdspy import formatters

def test_as_int():
    assert formatters.as_int(42, {}) == "42"
    assert formatters.as_int("123", {}) == "123"

def test_as_multiline():
    assert formatters.as_multiline("Hello\nWorld", {}) == "«Hello\nWorld»"

def test_as_list():
    assert formatters.as_list(["apple", "banana", "cherry"], {}) == "\n[0] apple[1] banana[2] cherry\n"

def test_as_json():
    data = {"name": "John", "age": 30}
    expected = '\n«{\n    "name": "John",\n    "age": 30\n}»\n'
    print(data)
    print(expected)
    print(formatters.as_json(data, {}))
    assert formatters.as_json(data, {}) == expected

def test_as_bulleted_list():
    items = ["apple", "banana", "cherry"]
    expected = '\n- apple\n- banana\n- cherry'
    assert formatters.as_bulleted_list(items, {}) == expected

def test_as_yaml():
    data = {"name": "John", "age": 30}
    expected = '\n«age: 30\nname: John\n»\n'
    print(data)
    print(expected)
    print(formatters.as_yaml(data, {}))
    assert formatters.as_yaml(data, {}) == expected