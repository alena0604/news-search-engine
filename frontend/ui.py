from io import BytesIO

import requests
import streamlit as st
from PIL import Image


class ImageHandler:
    """Handles downloading and resizing article images."""

    @staticmethod
    def download_image(url: str):
        """Download image from a URL."""
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
        except Exception as e:
            st.error(f"Error downloading image: {e}")
        return None

    @staticmethod
    def resize_image(image: Image.Image, size: tuple = (200, 150)):
        """Resize image to a shorter format."""
        return image.resize(size)


class ArticleRenderer:
    """Responsible for rendering search results on the Streamlit page."""

    @staticmethod
    def gradient_title(title: str):
        """Renders the article title with gradient styling."""
        html_title = f"""
        <h2 style='background: -webkit-linear-gradient(left, #ff7e5f, #feb47b);
        -webkit-background-clip: text; color: transparent;
        font-family: Verdana, sans-serif; font-size: 18px; font-weight: bold;
        margin-bottom: 10px;'>
        {title}</h2>
        """
        st.markdown(html_title, unsafe_allow_html=True)

    @staticmethod
    def render_article(article: dict):
        """Renders a single article card in Streamlit."""
        with st.container():
            image = ImageHandler.download_image(article["image_url"])
            if image:
                image = ImageHandler.resize_image(image)
                st.image(image, use_column_width=True, caption=article["title"])

            ArticleRenderer.gradient_title(article["title"])
            st.caption(
                f"<b>{article['date']}</b> &nbsp; | &nbsp; "
                f"<span style='color:green;'>Score: {100 * article['score']:.2f}%</span>",
                unsafe_allow_html=True,
            )

            # Render "See More" button
            st.markdown(
                f"""
                <a href="{article['url']}" target="_blank">
                    <button style="
                    background-color: #4CAF50; color: white;
                    padding: 10px 24px; border-radius: 12px; border: none;
                    font-size: 16px; transition: 0.3s;"
                    onmouseover="this.style.backgroundColor='#45a049'"
                    onmouseout="this.style.backgroundColor='#4CAF50'">
                    See More
                    </button>
                </a>
                """,
                unsafe_allow_html=True,
            )

    @staticmethod
    def display_articles(articles: list, columns: int = 2):
        """Displays a grid of article cards."""
        n_rows = (len(articles) + columns - 1) // columns
        for row in range(n_rows):
            cols = st.columns(columns)
            for col_idx in range(columns):
                idx = row * columns + col_idx
                if idx >= len(articles):
                    break
                with cols[col_idx]:
                    ArticleRenderer.render_article(articles[idx])
