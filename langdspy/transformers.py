from typing import Any, Dict, List, Type, Optional, Callable
import json
from enum import Enum
from langchain_core.documents import Document
import re
from .data_helper import normalize_enum_value

def as_bool(value: str, kwargs: Dict[str, Any]) -> bool:
    value = re.sub(r'[^\w\s]', '', value)
    value_parts = value.lower().split()
    return value_parts[0] in ("true", "yes", "1")

    
def as_json_list(val: str, kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    return json.loads(val)

def as_json(val: str, kwargs: Dict[str, Any]) -> Any:
    return json.loads(val)

def as_enum(val: str, kwargs: Dict[str, Any]) -> Enum:
    enum_class = kwargs['enum']
    normalized_val = normalize_enum_value(val)
    for member in enum_class:
        if normalize_enum_value(member.name) == normalized_val:
            return member
    raise ValueError(f"{val} is not a valid member of the {enum_class.__name__} enumeration")

def as_enum_list(val: str, kwargs: Dict[str, Any]) -> List[Enum]:
    enum_class = kwargs['enum']
    values = [v.strip() for v in val.split(",")]
    result = []
    for v in values:
        normalized_val = normalize_enum_value(v)
        for member in enum_class:
            if normalize_enum_value(member.name) == normalized_val:
                result.append(member)
                break
        else:
            raise ValueError(f"{v} is not a valid member of the {enum_class.__name__} enumeration")
    return result
