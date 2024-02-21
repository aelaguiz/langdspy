from typing import Any, Dict, List, Type, Optional, Callable
import yaml

import json
from langchain_core.documents import Document

def as_docs(docs: List[Document]) -> str:
    formatted_docs = ""
    for i, doc in enumerate(docs):
        formatted_docs += f"[{i}]«{doc.page_content}»\n"

    return formatted_docs

def as_multiline(input) -> str:
    return f"«{input}»"

def as_list(strings: List[str]) -> str:
    formatted_docs = ""
    for i, val in enumerate(strings):
        formatted_docs += f"[{i}] {val}"

    return '\n' + formatted_docs + '\n'

def as_json(obj: Dict[str, Any]) -> str:
    return '\n' + f"«{json.dumps(obj, indent=4)}»" + '\n'

def as_bulleted_list(items: List[str]) -> str:
    return '\n' + '\n'.join(f"- {item}" for item in items)

def as_yaml(obj: Any) -> str:
    return f"\n«{yaml.dump(obj, default_flow_style=False)}»\n"