
import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "password", "webscr", "ebayisapi", "paypal", "suspend",
    "recover", "unlock", "billing", "invoice", "security", "alert", "wallet",
    "gift", "bonus", "urgent", "limited",
]

SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "shorte.st", "cutt.ly", "rb.gy", "rebrand.ly", "tiny.cc",
]

SUSPICIOUS_TLDS = [
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "loan",
    "win", "review", "party", "stream", "gdn", "icu", "cam",
]

IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")

FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "num_dots", "num_hyphens",
    "num_at", "num_underscore", "num_percent", "num_query_params", "num_digits",
    "digit_ratio", "num_subdomains", "has_ip", "has_https", "has_port",
    "has_double_slash_redirect", "is_shortened", "suspicious_word_count",
    "has_suspicious_tld", "num_slashes_path",
]
FEATURE_DESCRIPTIONS = {
    "url_length": "Overall length of the URL",
    "hostname_length": "Length of the domain/host portion",
    "path_length": "Length of the page path after the domain",
    "num_dots": "Number of dots ( . ) in the URL",
    "num_hyphens": "Number of hyphens ( - ) in the URL",
    "num_at": "Presence of an @ symbol (can hide the real destination)",
    "num_underscore": "Number of underscores in the URL",
    "num_percent": "Number of percent-encoded characters (%xx)",
    "num_query_params": "Number of query-string parameters",
    "num_digits": "Number of digits in the URL",
    "digit_ratio": "Proportion of the URL made up of digits",
    "num_subdomains": "Number of subdomain levels before the main domain",
    "has_ip": "Domain is a raw IP address instead of a name",
    "has_https": "Connection uses HTTPS (encrypted)",
    "has_port": "URL specifies a non-standard port number",
    "has_double_slash_redirect": "Contains a // later in the URL (possible redirect trick)",
    "is_shortened": "Uses a known URL-shortening service",
    "suspicious_word_count": "Count of urgency/credential-themed words (login, verify, secure, ...)",
    "has_suspicious_tld": "Domain ending (TLD) is one commonly abused for phishing",
    "num_slashes_path": "Number of slashes in the page path",
}


def extract_features(url: str) -> dict:
    """Return a dict of {feature_name: value} for the given URL string."""
    url = (url or "").strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url):
        url_full = "http://" + url
    else:
        url_full = url

    try:
        parsed = urlparse(url_full)
    except ValueError:
        parsed = urlparse("http://invalid")

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    f = {}
    f["url_length"] = len(url)
    f["hostname_length"] = len(hostname)
    f["path_length"] = len(path)
    f["num_dots"] = url.count(".")
    f["num_hyphens"] = url.count("-")
    f["num_at"] = url.count("@")
    f["num_underscore"] = url.count("_")
    f["num_percent"] = url.count("%")
    f["num_query_params"] = (query.count("&") + 1) if query else 0

    digits = sum(c.isdigit() for c in url)
    f["num_digits"] = digits
    f["digit_ratio"] = round(digits / len(url), 4) if len(url) else 0.0

    hostname_parts = [p for p in hostname.split(".") if p] if hostname else []
    f["num_subdomains"] = max(len(hostname_parts) - 2, 0)
    f["has_ip"] = 1 if IP_PATTERN.match(hostname) else 0
    f["has_https"] = 1 if parsed.scheme == "https" else 0
    f["has_port"] = 1 if parsed.port else 0

    after_protocol = url_full.split("://", 1)[-1]
    f["has_double_slash_redirect"] = 1 if "//" in after_protocol else 0
    f["is_shortened"] = 1 if any(s in hostname for s in SHORTENERS) else 0

    low = url.lower()
    f["suspicious_word_count"] = sum(1 for w in SUSPICIOUS_WORDS if w in low)

    tld = hostname_parts[-1].lower() if hostname_parts else ""
    f["has_suspicious_tld"] = 1 if tld in SUSPICIOUS_TLDS else 0
    f["num_slashes_path"] = path.count("/")

    return f


def features_to_vector(feat_dict: dict) -> list:
    """Convert a feature dict into the ordered vector the model expects."""
    return [feat_dict[name] for name in FEATURE_NAMES]


if __name__ == "__main__":
    # quick manual smoke test
    tests = [
        "https://www.google.com",
        "http://192.168.1.1/login/verify-account.php?user=1&id=2",
        "http://secure-paypal-login.tk/update-account_info.php",
        "https://bit.ly/3xYzAbC",
    ]
    for t in tests:
        print(t)
        print(extract_features(t))
        print()
