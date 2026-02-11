from langchain_openai import ChatOpenAI
from src.config.settings import settings
from langchain_tavily import TavilySearch

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_retries=5,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
    base_url="https://openai.api.proxyapi.ru/v1",
)

tavily = TavilySearch(
    tavily_api_key=settings.TAVILY_API_KEY.get_secret_value(),
    max_results=5
)