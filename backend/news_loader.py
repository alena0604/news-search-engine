import datetime
import functools
import logging
from typing import Callable, List
from newsapi import NewsApiClient
from newsdataapi import NewsDataApiClient
from pydantic import ValidationError
from backend.models import CommonDocument, NewsAPIModel, NewsDataIOModel
from backend.settings import AppConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def error_handler(fetch_func: Callable) -> Callable:
    """
    It logs both expected validation errors and unexpected exceptions.

    Args:
        fetch_func (Callable): The wrapped API fetch function.

    Returns:
        Callable: The wrapped function with enhanced error handling.
    """

    @functools.wraps(fetch_func)
    def wrapped_function(*args, **kwargs):
        try:
            return fetch_func(*args, **kwargs)
        except ValidationError as validation_err:
            log.error(f"Validation error in {fetch_func.__name__}: {validation_err}")
        except Exception as err:
            log.error(f"Unexpected error in {fetch_func.__name__}: {err}")
            log.exception("Stack Trace:", exc_info=True)
        return []

    return wrapped_function


def get_time_interval(hours: int) -> str:
    """
    Generate a time window in ISO 8601 format for filtering news articles.

    Args:
        hours (int): The duration of the time window in hours.

    Returns:
        str: The start and end time as formatted strings.
    """
    now = datetime.datetime.utcnow()
    start_time = now - datetime.timedelta(hours=hours)
    return start_time.strftime("%Y-%m-%dT%H:%M:%S"), now.strftime("%Y-%m-%dT%H:%M:%S")


class ArticleFetcher:
    """
    A robust, modular class for fetching news articles from multiple APIs (NewsAPI, NewsDataAPI).

    Attributes:
        _newsapi (NewsApiClient): Client for NewsAPI.
        _newsdata (NewsDataApiClient): Client for NewsDataAPI.
        _window_hours (int): Time window (in hours) for retrieving news articles.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._newsapi = NewsApiClient(api_key=self.config.get("NEWSAPI_KEY"))
        self._newsdata = NewsDataApiClient(apikey=self.config.get("NEWSDATAIO_KEY"))
        self._window_hours = 24

    @error_handler
    def fetch_from_newsapi(self) -> List[CommonDocument]:
        """
        Fetches articles from the NewsAPI service with enhanced logging and validation.

        Returns:
            List[CommonDocument]: A list of articles transformed into the common document format.
        """
        try:
            log.debug(
                f"Fetching articles from NewsAPI with query: {self.config.get('NEWS_TOPIC')}"
            )
            response = self._newsapi.get_everything(
                q=self.config.get("NEWS_TOPIC"),
                language="en",
                page=self.config.get("ARTICLES_BATCH_SIZE", 5),
                page_size=self.config.get("ARTICLES_BATCH_SIZE", 5),
            )
            articles = response.get("articles", [])
            log.info(f"Fetched {len(articles)} articles from NewsAPI.")
            return [NewsAPIModel(**item).to_common() for item in articles]
        except Exception as ex:
            log.error(f"Error during NewsAPI fetch: {ex}")
            return []

    @error_handler
    def fetch_from_newsdataapi(self) -> List[CommonDocument]:
        """
        Fetches articles from the NewsDataAPI service, handling errors and response parsing.

        Returns:
            List[CommonDocument]: A list of articles transformed into the common document format.
        """
        try:
            log.debug(
                f"Fetching articles from NewsDataAPI with query: {self.config.get('NEWS_TOPIC')}"
            )
            response = self._newsdata.latest_api(
                q=self.config.get("NEWS_TOPIC"),
                language="en",
                size=self.config.get("ARTICLES_BATCH_SIZE", 5),
            )
            articles = response.get("results", [])
            log.info(f"Fetched {len(articles)} articles from NewsDataAPI.")
            return [NewsDataIOModel(**item).to_common() for item in articles]
        except Exception as ex:
            log.error(f"Error during NewsDataAPI fetch: {ex}")
            return []

    @property
    def sources(self) -> List[Callable]:
        """
        Retrieves the list of available news-fetching functions for parallel or sequential execution.

        Returns:
            List[Callable]: A list of fetch functions from NewsAPI and NewsDataAPI.
        """
        log.info("Compiling list of news sources for fetching.")
        return [self.fetch_from_newsapi, self.fetch_from_newsdataapi]

    def fetch_all_sources(self) -> List[CommonDocument]:
        """
        Fetches articles from all sources and aggregates the results, ensuring no duplicates.

        Returns:
            List[CommonDocument]: Aggregated list of news articles.
        """
        all_articles = []
        for fetch_func in self.sources:
            articles = fetch_func()
            if articles:
                all_articles.extend(articles)

        unique_articles = self._remove_duplicates(all_articles)
        log.info(f"Total unique articles fetched: {len(unique_articles)}")
        return unique_articles

    def _remove_duplicates(
        self, articles: List[CommonDocument]
    ) -> List[CommonDocument]:
        """
        Removes duplicate articles based on their unique URL or ID.

        Args:
            articles (List[CommonDocument]): List of articles to process.

        Returns:
            List[CommonDocument]: Deduplicated list of articles.
        """
        seen_urls = set()
        unique_articles = []
        for article in articles:
            if article.url not in seen_urls:
                unique_articles.append(article)
                seen_urls.add(article.url)
        return unique_articles
