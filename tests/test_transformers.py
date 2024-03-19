# tests/test_transformers.py
import pytest
import json
from langdspy import transformers
from enum import Enum

def test_as_bool():
    assert transformers.as_bool("true", {}) == True
    assert transformers.as_bool("yes", {}) == True
    assert transformers.as_bool("1", {}) == True
    assert transformers.as_bool("false", {}) == False
    assert transformers.as_bool("no", {}) == False
    assert transformers.as_bool("0", {}) == False

def test_as_json_list():
    assert transformers.as_json_list('["apple", "banana"]', {}) == ["apple", "banana"]
    
    with pytest.raises(json.JSONDecodeError):
        transformers.as_json_list('invalid json', {})

def test_as_enum():
    class Fruit(Enum):
        APPLE = 1
        BANANA = 2
    
    assert transformers.as_enum("APPLE", {"enum": Fruit}) == Fruit.APPLE
    assert transformers.as_enum("BANANA", {"enum": Fruit}) == Fruit.BANANA
    
    with pytest.raises(ValueError):
        transformers.as_enum("CHERRY", {"enum": Fruit})

def test_as_enum_list():
    class Fruit(Enum):
        APPLE = 1
        BANANA = 2
        CHERRY = 3
    
    assert transformers.as_enum_list("APPLE", {"enum": Fruit}) == [Fruit.APPLE]
    assert transformers.as_enum_list("BANANA, CHERRY", {"enum": Fruit}) == [Fruit.BANANA, Fruit.CHERRY]
    assert transformers.as_enum_list("APPLE,BANANA,CHERRY", {"enum": Fruit}) == [Fruit.APPLE, Fruit.BANANA, Fruit.CHERRY]
    
    with pytest.raises(ValueError):
        transformers.as_enum_list("DURIAN", {"enum": Fruit})