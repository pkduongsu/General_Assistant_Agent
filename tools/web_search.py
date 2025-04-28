from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return up to 3 results as <Document/> blocks."""
    search_tool = TavilySearchResults(max_results=3)
    # Unpack the (results_list, raw_response_dict) tuple
    results = search_tool.invoke(input=query)  # :contentReference[oaicite:0]{index=0}

    formatted = []
    for item in results:
        url     = item.get("url", "")
        title   = item.get("title", "")
        # choose either the short 'content' or the full 'raw_content'
        snippet = item.get("content", "") or item.get("raw_content", "")
        formatted.append(
            f'<Document source="{url}" title="{title}">\n'
            f'{snippet}\n'
            f'</Document>'
        )

    return "\n\n---\n\n".join(formatted)



if __name__ == "__main__":
    query = "Python programming language"
    # call via .invoke(input=...) since @tool wraps it as a BaseTool
    result = web_search.invoke(input=query)
    print(result)
