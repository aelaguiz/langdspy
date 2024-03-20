# tests/test_validators.py
import pytest
from langdspy import validators

def test_is_json_list():
    assert validators.is_json_list({}, '["apple", "banana"]', {}) == True
    assert validators.is_json_list({}, '{"name": "John"}', {}) == False
    assert validators.is_json_list({}, 'invalid json', {}) == False

def test_is_one_of():
    assert validators.is_one_of({}, 'apple', {'choices': ['apple', 'banana']}) == True
    assert validators.is_one_of({}, 'cherry', {'choices': ['apple', 'banana']}) == False
    assert validators.is_one_of({}, 'APPLE', {'choices': ['apple', 'banana'], 'case_sensitive': False}) == True
    assert validators.is_one_of({}, 'none', {'choices': ['apple', 'banana'], 'none_ok': True}) == True
    assert validators.is_one_of({}, 'apple pie', {'choices': ['apple_pie', 'banana split'], 'case_sensitive': False}) == True
    assert validators.is_one_of({}, 'Apple-Pie', {'choices': ['apple_pie', 'banana-split'], 'case_sensitive': False}) == True
    
    with pytest.raises(ValueError):
        validators.is_one_of({}, 'apple', {})

def test_is_subset_of():
    choices = ["apple", "banana", "cherry_pie", "durian-fruit"]
    
    assert validators.is_subset_of({}, "apple", {"choices": choices}) == True
    assert validators.is_subset_of({}, "apple,banana", {"choices": choices}) == True
    assert validators.is_subset_of({}, "apple, banana, cherry_pie", {"choices": choices}) == True
    assert validators.is_subset_of({}, "APPLE", {"choices": choices, "case_sensitive": False}) == True
    assert validators.is_subset_of({}, "APPLE,BANANA", {"choices": choices, "case_sensitive": False}) == True
    assert validators.is_subset_of({}, "Durian-Fruit, Cherry Pie", {"choices": choices, "case_sensitive": False}) == True
    
    assert validators.is_subset_of({}, "mango", {"choices": choices}) == False
    assert validators.is_subset_of({}, "apple,mango", {"choices": choices}) == False
    
    assert validators.is_subset_of({}, "none", {"choices": choices, "none_ok": True}) == True
    assert validators.is_subset_of({}, "apple,none", {"choices": choices, "none_ok": True}) == False
    
    with pytest.raises(ValueError):
        validators.is_subset_of({}, "apple", {})