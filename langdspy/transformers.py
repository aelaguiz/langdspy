from typing import Any, Dict, List, Type, Optional, Callable
import json
from enum import Enum
from langchain_core.documents import Document
import re

def as_bool(value: str, kwargs: Dict[str, Any]) -> bool:
    value = re.sub(r'[^\w\s]', '', value)
    value_parts = value.lower().split()
    return value_parts[0] in ("true", "yes", "1")

    
def as_json_list(val: str, kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    return json.loads(val)

def as_enum(val: str, kwargs: Dict[str, Any]) -> Enum:
    print(f"Transforming {val} to enum")
    enum_definition = kwargs['enum']

def as_enum(val: str, kwargs: Dict[str, Any]) -> Enum:
    enum_class = kwargs['enum']
    try:
        return enum_class[val.upper()]
    except KeyError:
        raise ValueError(f"{val} is not a valid member of the {enum_class.__name__} enumeration")