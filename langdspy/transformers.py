from typing import Any, Dict, List, Type, Optional, Callable
import json
from langchain_core.documents import Document
import re

def as_bool(value: str) -> bool:
    value = re.sub(r'[^\w\s]', '', value)
    value_parts = value.lower().split()
    return value_parts[0] in ("true", "yes", "1")

    
def as_json_list(val: str) -> List[Dict[str, Any]]:
    return json.loads(val)