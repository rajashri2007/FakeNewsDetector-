from source_verification import normalize_domain, validate_url, verify_source


def test_validate_url_accepts_http_url():
    assert validate_url("https://www.bbc.com/news") is True


def test_validate_url_rejects_bad_input():
    assert validate_url("not a url") is False
    assert validate_url("") is False


def test_normalize_domain_removes_scheme_and_www():
    assert normalize_domain("https://www.reuters.com/world") == "reuters.com"


def test_known_source_returns_expected_status():
    result = verify_source("bbc.com")
    assert result["status"] == "Known publisher"
    assert result["domain"] == "bbc.com"


def test_unknown_source_returns_not_verified():
    result = verify_source("some-random-source-example.org")
    assert result["status"] == "Source not verified"
