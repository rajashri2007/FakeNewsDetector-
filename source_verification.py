import json
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DATABASE_PATH = BASE_DIR / "data" / "trusted_sources.json"


def normalize_domain(value):
    """Return a cleaned domain name without scheme or www."""
    if value is None:
        return ""

    text = str(value).strip().lower()
    if not text:
        return ""

    if "://" not in text:
        text = f"https://{text}"

    parsed = urlparse(text)
    domain = (parsed.netloc or parsed.path or "").replace("www.", "")
    return domain.split("/")[0].strip()


def validate_url(raw_url):
    """Check whether the URL is valid and uses HTTP/HTTPS."""
    if raw_url is None:
        return False

    value = str(raw_url).strip()
    if not value:
        return False

    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def load_trusted_sources():
    """Read trusted sources from the configured JSON list."""
    if not SOURCE_DATABASE_PATH.exists():
        return []

    try:
        with SOURCE_DATABASE_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return []

    return data if isinstance(data, list) else []


def verify_source(domain):
    """Return a simple status for a given domain."""
    normalized = normalize_domain(domain)

    if not normalized:
        return {
            "status": "Unable to verify source",
            "publisher": "Unknown",
            "domain": "",
            "category": "unknown",
        }

    sources = load_trusted_sources()

    for source in sources:
        if normalize_domain(source.get("domain")) == normalized:
            return {
                "status": "Known publisher",
                "publisher": source.get("publisher_name", "Unknown"),
                "domain": normalized,
                "category": source.get("category", "unknown"),
            }

    if not sources:
        return {
            "status": "Unable to verify source",
            "publisher": "Unknown",
            "domain": normalized,
            "category": "unknown",
        }

    return {
        "status": "Source not verified",
        "publisher": "Unknown",
        "domain": normalized,
        "category": "unknown",
    }


def download_webpage(url, timeout=10):
    """Download a webpage safely with a timeout and raise clear errors."""
    if not validate_url(url):
        raise ValueError("Invalid or empty URL was provided.")

    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        raise ValueError(f"The URL could not be accessed: {exc}") from exc


def extract_article_metadata(url):
    """Fetch a webpage, extract title/text metadata, and return a dictionary."""
    response = download_webpage(url)
    soup = BeautifulSoup(response.text, "html.parser")

    title = ""

    meta_title = soup.select_one('meta[property="og:title"]')
    if meta_title and meta_title.get("content"):
        title = meta_title["content"].strip()

    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(" ", strip=True)

    if not title:
        heading = soup.find("h1")
        if heading:
            title = heading.get_text(" ", strip=True)

    author = ""
    author_candidates = [
        soup.select_one('meta[name="author"]'),
        soup.select_one('meta[property="article:author"]'),
        soup.select_one('[rel="author"]'),
        soup.select_one(".author"),
    ]

    for candidate in author_candidates:
        if candidate:
            author = candidate.get("content") or candidate.get_text(" ", strip=True)
            if author:
                break

    publication_date = ""
    date_candidates = [
        soup.select_one('meta[property="article:published_time"]'),
        soup.select_one('meta[name="pubdate"]'),
        soup.select_one('meta[name="publishdate"]'),
        soup.select_one('time'),
    ]

    for candidate in date_candidates:
        if candidate:
            publication_date = candidate.get("content") or candidate.get_text(" ", strip=True)
            if publication_date:
                break

    paragraphs = soup.find_all("p")
    article_text = " ".join(
        paragraph.get_text(" ", strip=True)
        for paragraph in paragraphs
        if paragraph.get_text(" ", strip=True)
    )

    if len(article_text) > 3000:
        article_text = article_text[:3000]

    domain = normalize_domain(url)

    return {
        "title": title,
        "text": article_text,
        "domain": domain,
        "author": author,
        "publication_date": publication_date,
        "source_status": verify_source(domain)["status"],
    }


if __name__ == "__main__":
    print(verify_source("https://www.bbc.com"))
