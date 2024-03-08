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
    
    with pytest.raises(ValueError):
        validators.is_one_of({}, 'apple', {})