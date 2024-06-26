from typing import Any, Dict, List, Type, Optional, Callable
import yaml

import json
from langchain_core.documents import Document

def as_docs(docs: List[Document], kwargs: Dict) -> str:
    max_doc_length = kwargs.get("max_doc_length", None)
    formatted_docs = ""
    for i, doc in enumerate(docs):
        content = doc.page_content

        if len(content) > max_doc_length:
            content = content[:max_doc_length] + "..."
        formatted_docs += f"[{i}]«{content}»\n"

    return formatted_docs

def as_int(input, kwargs: Dict) -> str:
    return f"{input}"

def as_multiline(input, kwargs: Dict) -> str:
    return f"«{input}»"

def as_list(strings: List[str], kwargs: Dict) -> str:
    formatted_docs = ""
    for i, val in enumerate(strings):
        formatted_docs += f"[{i}] {val}"

    return '\n' + formatted_docs + '\n'

def as_json(obj: Dict[str, Any], kwargs: Dict) -> str:
    return '\n' + f"«{json.dumps(obj, indent=4)}»" + '\n'

def as_bulleted_list(items: List[str], kwargs: Dict) -> str:
    return '\n' + '\n'.join(f"- {item}" for item in items)

def as_yaml(obj: Any, kwargs: Dict) -> str:
    print(f"Formatting as yaml: {obj}")
    return f"\n«{yaml.dump(obj, default_flow_style=False)}»\n"