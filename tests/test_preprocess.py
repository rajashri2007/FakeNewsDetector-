from preprocess import clean_text


def test_clean_text_removes_html_and_urls():
    raw_text = "<p>Breaking news! The system is <b>real</b> and here: https://example.com</p>"
    cleaned = clean_text(raw_text)

    assert "breaking" in cleaned
    assert "real" in cleaned
    assert "https" not in cleaned
    assert "example" not in cleaned


def test_clean_text_handles_empty_input():
    cleaned = clean_text("")
    assert cleaned == ""
