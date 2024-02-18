from typing import Any, Dict, List, Type, Optional, Callable
from langchain_core.documents import Document

def as_bool(value: str) -> bool:
    value_parts = value.lower().split()

    return value_parts[0] in ("true", "yes", "1")