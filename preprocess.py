import re
from typing import Optional

from bs4 import BeautifulSoup


def clean_text(raw_text: Optional[str]) -> str:
    """Clean text while preserving Unicode/Indian-language characters."""

    if raw_text is None:
        return ""

    text = str(raw_text)

    # Remove HTML tags but keep readable text
    text = BeautifulSoup(text, "html.parser").get_text(" ")

    # Remove URLs and emails
    text = re.sub(
        r"https?://\S+|www\.\S+|[\w\.-]+@[\w\.-]+",
        " ",
        text
    )

    # Keep Unicode letters/numbers and spaces
    # This preserves Hindi, Marathi, English, etc.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Convert English letters to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    sample = "भारत सरकार ने नया कार्यक्रम शुरू किया। Visit http://example.com"

    print("Original:")
    print(sample)

    print("\nCleaned:")
    print(clean_text(sample))