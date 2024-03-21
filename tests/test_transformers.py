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
        CHERRY_PIE = 3
        DURIAN_FRUIT = 4
    
    assert transformers.as_enum("APPLE", {"enum": Fruit}) == Fruit.APPLE
    assert transformers.as_enum("BANANA", {"enum": Fruit}) == Fruit.BANANA
    assert transformers.as_enum("cherry pie", {"enum": Fruit}) == Fruit.CHERRY_PIE
    assert transformers.as_enum("Durian-Fruit", {"enum": Fruit}) == Fruit.DURIAN_FRUIT
    assert transformers.as_enum("Durian_Fruit", {"enum": Fruit}) == Fruit.DURIAN_FRUIT
    assert transformers.as_enum("Durian Fruit", {"enum": Fruit}) == Fruit.DURIAN_FRUIT
    
    with pytest.raises(ValueError):
        transformers.as_enum("MANGO", {"enum": Fruit})

def test_as_enum_list():
    class Fruit(Enum):
        APPLE = 1
        BANANA = 2
        CHERRY_PIE = 3
        DURIAN_FRUIT = 4
    
    assert transformers.as_enum_list("APPLE", {"enum": Fruit}) == [Fruit.APPLE]
    assert transformers.as_enum_list("BANANA, CHERRY PIE", {"enum": Fruit}) == [Fruit.BANANA, Fruit.CHERRY_PIE]
    assert transformers.as_enum_list("APPLE,BANANA,CHERRY PIE", {"enum": Fruit}) == [Fruit.APPLE, Fruit.BANANA, Fruit.CHERRY_PIE]
    assert transformers.as_enum_list("Durian-Fruit, cherry pie", {"enum": Fruit}) == [Fruit.DURIAN_FRUIT, Fruit.CHERRY_PIE]
    assert transformers.as_enum_list("Durian Fruit, cherry_pie", {"enum": Fruit}) == [Fruit.DURIAN_FRUIT, Fruit.CHERRY_PIE]
    
    with pytest.raises(ValueError):
        transformers.as_enum_list("MANGO", {"enum": Fruit})