import os
from print_utils import print
from tavily import TavilyClient
from typing import List

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def fetch_recent_news(query: str) -> List[str]:
    """
    Inputs a query string, searches for news, and returns the top results.

    Args:
    query: String to search

    Returns:
    content: List of strings, each containing a news article about the topic
    """
    response = tavily_client.search(query, topic="news", max_results=4)
    return [x["content"] for x in response["results"]]



if __name__ == "__main__":
    responses = fetch_recent_news("Kimi model")
    print(responses)



