from unstructured.cleaners.core import (
    clean,
    clean_non_ascii_chars,
    replace_unicode_quotes,
    remove_punctuation,
)
import re


def remove_html_tags(text):
    html_tag_pattern = r"<[^>]+>"
    return re.sub(html_tag_pattern, "", text)


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def clean_full(text: str) -> str:
    """
    Cleans the given text by applying the following set of operations:
    - clean (e.g. whitespaces)
    - clean_ordered_bullets (e.g. bullets)
    - replace_unicode_quotes (e.g. quotes)
    - clean_non_ascii_chars (e.g. non-ascii characters)
    - remove_punctuation (e.g. punctuation)

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = clean(
        text=text, lowercase=True, extra_whitespace=True, dashes=True, bullets=True
    )
    text = replace_unicode_quotes(text)
    text = clean_non_ascii_chars(text)
    text = remove_punctuation(text)
    return text
