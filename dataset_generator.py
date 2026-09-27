
import numpy as np
import pandas as pd

from feature_extraction import FEATURE_NAMES


def _clip_nonneg(arr):
    return np.clip(arr, 0, None)


def generate_dataset(n: int = 9000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_phish = n // 2
    n_legit = n - n_phish

    rows = []

    # ---- Phishing-like samples ----
    for _ in range(n_phish):
        url_length = _clip_nonneg(rng.normal(70, 25))
        hostname_length = _clip_nonneg(rng.normal(24, 8))
        path_length = _clip_nonneg(rng.normal(28, 15))
        num_dots = _clip_nonneg(rng.normal(3.2, 1.5))
        num_hyphens = _clip_nonneg(rng.poisson(2.2))
        num_at = rng.choice([0, 1], p=[0.92, 0.08])
        num_underscore = _clip_nonneg(rng.poisson(0.6))
        num_percent = _clip_nonneg(rng.poisson(0.5))
        num_query_params = _clip_nonneg(rng.poisson(1.0))
        digit_ratio = np.clip(rng.normal(0.12, 0.08), 0, 1)
        num_digits = _clip_nonneg(url_length * digit_ratio)
        num_subdomains = _clip_nonneg(rng.poisson(1.4))
        has_ip = rng.choice([0, 1], p=[0.78, 0.22])
        has_https = rng.choice([0, 1], p=[0.55, 0.45])
        has_port = rng.choice([0, 1], p=[0.95, 0.05])
        has_double_slash_redirect = rng.choice([0, 1], p=[0.85, 0.15])
        is_shortened = rng.choice([0, 1], p=[0.82, 0.18])
        suspicious_word_count = _clip_nonneg(rng.poisson(1.8))
        has_suspicious_tld = rng.choice([0, 1], p=[0.6, 0.4])
        num_slashes_path = _clip_nonneg(rng.poisson(2.0))

        rows.append([
            url_length, hostname_length, path_length, num_dots, num_hyphens,
            num_at, num_underscore, num_percent, num_query_params, num_digits,
            round(digit_ratio, 4), num_subdomains, has_ip, has_https, has_port,
            has_double_slash_redirect, is_shortened, suspicious_word_count,
            has_suspicious_tld, num_slashes_path, 1,
        ])

    # ---- Legitimate-like samples ----
    for _ in range(n_legit):
        url_length = _clip_nonneg(rng.normal(32, 12))
        hostname_length = _clip_nonneg(rng.normal(15, 5))
        path_length = _clip_nonneg(rng.normal(10, 8))
        num_dots = _clip_nonneg(rng.normal(1.8, 0.7))
        num_hyphens = _clip_nonneg(rng.poisson(0.4))
        num_at = rng.choice([0, 1], p=[0.995, 0.005])
        num_underscore = _clip_nonneg(rng.poisson(0.1))
        num_percent = _clip_nonneg(rng.poisson(0.05))
        num_query_params = _clip_nonneg(rng.poisson(0.3))
        digit_ratio = np.clip(rng.normal(0.02, 0.03), 0, 1)
        num_digits = _clip_nonneg(url_length * digit_ratio)
        num_subdomains = _clip_nonneg(rng.poisson(0.4))
        has_ip = rng.choice([0, 1], p=[0.995, 0.005])
        has_https = rng.choice([0, 1], p=[0.08, 0.92])
        has_port = rng.choice([0, 1], p=[0.995, 0.005])
        has_double_slash_redirect = rng.choice([0, 1], p=[0.97, 0.03])
        is_shortened = rng.choice([0, 1], p=[0.96, 0.04])
        suspicious_word_count = _clip_nonneg(rng.poisson(0.15))
        has_suspicious_tld = rng.choice([0, 1], p=[0.95, 0.05])
        num_slashes_path = _clip_nonneg(rng.poisson(0.8))

        rows.append([
            url_length, hostname_length, path_length, num_dots, num_hyphens,
            num_at, num_underscore, num_percent, num_query_params, num_digits,
            round(digit_ratio, 4), num_subdomains, has_ip, has_https, has_port,
            has_double_slash_redirect, is_shortened, suspicious_word_count,
            has_suspicious_tld, num_slashes_path, 0,
        ])

    cols = FEATURE_NAMES + ["label"]
    df = pd.DataFrame(rows, columns=cols)
    int_cols = [c for c in FEATURE_NAMES if c != "digit_ratio"]
    df[int_cols] = df[int_cols].round().astype(int)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/synthetic_dataset.csv", index=False)
    print(df.shape)
    print(df.groupby("label").mean(numeric_only=True).T)
