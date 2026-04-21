from typing import Any, Dict

from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    raw = web_search_tool.invoke({"query": question})
    if not isinstance(raw, dict):
        joined_tavily_result = str(raw)
    elif "error" in raw:
        joined_tavily_result = f"Search error: {raw['error']}"
    else:
        parts: list[str] = []
        if answer := raw.get("answer"):
            parts.append(str(answer))
        for item in raw.get("results") or []:
            if isinstance(item, dict) and item.get("content"):
                parts.append(str(item["content"]))
            elif isinstance(item, dict):
                title = item.get("title", "")
                url = item.get("url", "")
                parts.append(f"{title}\n{url}".strip())
        joined_tavily_result = "\n\n".join(parts) if parts else ""

    web_results=Document(page_content=joined_tavily_result)

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents,"question": question}    
if __name__ == "__main__":
    web_search({"question": "agent memory", "documents": None})