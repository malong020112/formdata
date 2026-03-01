import json
import time
from typing import List, Union
from qwen_agent.tools.base import BaseTool
from typing import Optional
import http.client

from config import SERPER_KEY

SERPER_HOST = "google.serper.dev"
SERPER_TIMEOUT_SEC = 12
SERPER_MAX_RETRIES = 5


class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def google_search_with_serp(self, query: str):
        def contains_chinese_basic(text: str) -> bool:
            return any('\u4E00' <= char <= '\u9FFF' for char in text)

        if not SERPER_KEY:
            return "Search API key is empty (SERPER_KEY)."

        if contains_chinese_basic(query):
            payload = json.dumps({
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn"
            })
            
        else:
            payload = json.dumps({
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en"
            })
        headers = {
                'X-API-KEY': SERPER_KEY,
                'Content-Type': 'application/json'
            }

        last_error: Optional[Exception] = None
        results = None

        for i in range(SERPER_MAX_RETRIES):
            conn = http.client.HTTPSConnection(SERPER_HOST, timeout=SERPER_TIMEOUT_SEC)
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                raw = res.read().decode("utf-8", errors="replace")
                if res.status >= 400:
                    raise RuntimeError(f"HTTP {res.status}: {raw[:300]}")
                results = json.loads(raw)
                break
            except Exception as e:
                last_error = e
                print(e)
                if i == SERPER_MAX_RETRIES - 1:
                    return f"Google search failed after retries: {e}"
                # Exponential backoff: 1s, 2s, 4s, ...
                time.sleep(2 ** i)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        if results is None:
            return f"Google search failed: {last_error}"

        try:
            if "organic" not in results:
                return f"No results found for '{query}'. Try with a more general query."

            web_snippets = list()
            idx = 0
            if "organic" in results:
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

            content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            return content
        except:
            return f"No results found for '{query}'. Try with a more general query."


    
    def search_with_serp(self, query: str):
        result = self.google_search_with_serp(query)
        return result

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            # 单个查询
            response = self.search_with_serp(query)
        else:
            # 多个查询
            assert isinstance(query, List)
            responses = []
            for q in query:
                responses.append(self.search_with_serp(q))
            response = "\n=======\n".join(responses)
            
        return response
