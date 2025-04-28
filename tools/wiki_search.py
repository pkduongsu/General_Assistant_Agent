from langchain_community.document_loaders.wikipedia import WikipediaLoader
from langchain_core.tools import tool
from typing import List
from langchain.schema import Document


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return up to 2 results, formatted as <Document/> blocks.
    
    Args:
        query: The search query.
    """
    # load returns a List[Document]
    search_docs: List[Document] = WikipediaLoader(
        query=query,
        load_max_docs=2
    ).load()

    formatted_docs = []
    for doc in search_docs:
        # Document objects expose .metadata and .page_content
        src  = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        text = doc.page_content

        formatted_docs.append(
            f'<Document source="{src}" page="{page}">\n'
            f'{text}\n'
            f'</Document>'
        )

    return "\n\n---\n\n".join(formatted_docs)


if __name__ == "__main__":
    query = "Python programming language"

    # Since @tool turned wiki_search into a BaseTool,
    # call invoke(input=...) rather than calling it directly.
    result = wiki_search.invoke(input=query)
    print(result)
