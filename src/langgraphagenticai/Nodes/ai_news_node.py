import os
import hashlib
from datetime import datetime
from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate


class AINewsNode:
    def __init__(self, llm):
        """Initialize the AINewsNode with Tavily client and LLM"""
        self.tavily = TavilyClient()
        self.llm = llm
        self.state = {}

    def fetch_news(self, state: dict) -> dict:
        """Fetch AI news based on the specified frequency."""
        frequency = state['messages'][0].content.strip().lower()
        self.state['frequency'] = frequency

        time_range_map = {'daily': 'd', 'weekly': 'w', 'monthly': 'm', 'year': 'y'}
        days_map = {'daily': 1, 'weekly': 7, 'monthly': 30, 'year': 365}

        if frequency not in time_range_map:
            raise ValueError(f"Invalid frequency '{frequency}'. Choose from: {list(time_range_map.keys())}")

        response = self.tavily.search(
            query='Top AI news',
            topic='news',
            time_range=time_range_map[frequency],
            include_answer="advanced",
            max_results=20,
            days=days_map[frequency]
        )

        news_results = response.get('results', [])
        self.state['news_data'] = news_results
        state['news_data'] = news_results
        return state

    def summarize_news(self, state: dict) -> dict:
        """Summarize the fetched news using the LLM."""
        news_items = self.state.get('news_data', [])

        # Deduplicate based on title + content hash
        seen_hashes = set()
        unique_articles = []
        for item in news_items:
            title = item.get('title', '')
            content = item.get('content', '')
            url = item.get('url', '')
            if not url or not content:
                continue
            combined = f"{title}|{content}"
            content_hash = hashlib.md5(combined.encode('utf-8')).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_articles.append(item)

        # Sort by published date (newest first)
        def parse_date(item):
            try:
                return datetime.strptime(item.get('published_date', ''), "%Y-%m-%d")
            except Exception:
                return datetime.min

        unique_articles.sort(key=parse_date, reverse=True)
        articles_to_summarize = unique_articles[:15]

        if not articles_to_summarize:
            summary = "No AI news found for the selected time frame."
            state['summary'] = summary
            self.state['summary'] = summary
            return self.state

        # Prepare article string
        articles_str = "\n\n".join([
            f"Title: {item.get('title', '')}\n"
            f"Content: {item.get('content', '')}\n"
            f"Date: {item.get('published_date', '')}\n"
            f"URL: {item.get('url', '')}"
            for item in articles_to_summarize
        ])

        # Prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a professional summarizer.

Summarize each AI news article clearly and briefly in markdown format.
Use this structure:

### [YYYY-MM-DD]
- [1–2 sentence summary](URL)

Ensure summaries are concise, informative, and sorted by latest date first.
Only include useful, unique articles.
"""
            ),
            ("user", "Articles:\n\n{articles}")
        ])

        formatted_prompt = prompt_template.format(articles=articles_str)
        response = self.llm.invoke(formatted_prompt)

        summary = response.content
        state['summary'] = summary
        self.state['summary'] = summary
        return self.state

    def save_result(self, state: dict) -> dict:
        """Save the summarized news to a markdown file."""
        frequency = self.state['frequency']
        summary = self.state['summary']

        os.makedirs("./AINews", exist_ok=True)
        filename = f"./AINews/{frequency}_summary.md"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# {frequency.capitalize()} AI News Summary\n\n")
            f.write(summary)

        self.state['filename'] = filename
        return self.state
