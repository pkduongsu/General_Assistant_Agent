from langchain_community.document_loaders import ArxivLoader
from langchain_core.tools import tool
from typing import List

@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return up to 2 results, formatted as <Document/> blocks.
    
    Args:
        query: The search query.
    """
    # load returns a List[Document]
    search_docs: List[Document] = ArxivLoader(
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
    result = arxiv_search.invoke(input=query)
    print(result)