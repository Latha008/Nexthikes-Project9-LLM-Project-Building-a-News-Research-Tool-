"""
LangChain Configuration File
- This file handles summarization using OpenAI and LangChain
- News is fetched using NewsAPI
"""

# Initialize LLM and prompt
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from newsapi import NewsApiClient

openai_api_key = 'your-openai-key'  # Replace with your actual OpenAI API key
openai = OpenAI(api_api_key=openai_api_key)

# Prompt for summarization
template = """
You are an AI assistant helping an equity research analyst. Given the query and summaries, return a detailed summary.
Query: {query}
Summaries: {summaries}
"""
prompt = PromptTemplate(template=template, input_variables=["query", "summaries"])
llm_chain = LLMChain(prompt=prompt, llm=openai)

def get_news_articles(query, api_key="YOUR_NEWSAPI_KEY"): # Replace with your NewsAPI key
    """Fetches news articles based on the query using NewsAPI."""
    newsapi = NewsApiClient(api_key=api_key)
    try:
        top_headlines = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=10)
        articles = top_headlines.get('articles', [])
        return articles
    except Exception as e:
        print(f"Error fetching news articles: {e}")
        return []

def summarize_articles(articles):
    """Combines article descriptions into a single summary string."""
    summaries = [article.get('description') or article.get('content') for article in articles if article.get('description') or article.get('content')]
    # Simple concatenation for now, can be improved
    return "\n\n".join(summaries)

def get_summary(query):
    """Fetches news articles and generates a summary using the LLM chain."""
    articles = get_news_articles(query)
    summaries = summarize_articles(articles)
    if not summaries:
        return "Could not retrieve or summarize articles for the given query."
    return llm_chain.run({'query': query, 'summaries': summaries})
