from typing import Any, Dict, List, Type, Optional, Callable
from langchain_core.documents import Document

def doc_formatter(docs: List[Document]) -> str:
    formatted_docs = ""
    for i, doc in enumerate(docs):
        formatted_docs += f"[{i}]«\"\"\"{doc.page_content}\"\"\"»\n"

    return formatted_docs