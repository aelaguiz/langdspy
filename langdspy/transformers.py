from typing import Any, Dict, List, Type, Optional, Callable
from langchain_core.documents import Document
import re

def as_bool(value: str) -> bool:
    value = re.sub(r'[^\w\s]', '', value)
    value_parts = value.lower().split()
    return value_parts[0] in ("true", "yes", "1")