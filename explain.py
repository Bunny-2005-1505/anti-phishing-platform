
import json

from feature_extraction import FEATURE_DESCRIPTIONS, FEATURE_NAMES

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


def _load_json(path):
    with open(path) as f:
        return json.load(f)


class Explainer:
    def __init__(self, model, models_dir="models"):
        self.model = model
        self.importances = _load_json(f"{models_dir}/feature_importance.json")
        self.stats = _load_json(f"{models_dir}/train_stats.json")
        self._shap_explainer = None
        if _HAS_SHAP:
            try:
                self._shap_explainer = shap.TreeExplainer(model)
            except Exception:
                self._shap_explainer = None

    def explain(self, feature_dict: dict, top_k: int = 5) -> list:
        """Return a list of dicts, most-influential feature first:
        {feature, description, value, direction, sentence}
        direction is 'phishing' or 'legitimate'."""
        if self._shap_explainer is not None:
            return self._explain_shap(feature_dict, top_k)
        return self._explain_heuristic(feature_dict, top_k)

    # ---- SHAP path -----------------------------------------------------
    def _explain_shap(self, feature_dict, top_k):
        import numpy as np

        vec = np.array([[feature_dict[name] for name in FEATURE_NAMES]])
        shap_values = self._shap_explainer.shap_values(vec)
        # Binary classifiers: some sklearn/shap combos return a list [class0, class1]
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        pairs = list(zip(FEATURE_NAMES, sv))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)

        out = []
        for name, contrib in pairs[:top_k]:
            direction = "phishing" if contrib > 0 else "legitimate"
            out.append(self._format_entry(name, feature_dict[name], direction, abs(contrib)))
        return out

    # ---- Heuristic fallback path ---------------------------------------
    def _explain_heuristic(self, feature_dict, top_k):
        ranked = sorted(self.importances.items(), key=lambda kv: kv[1], reverse=True)

        scored = []
        for name, importance in ranked:
            s = self.stats[name]
            value = feature_dict[name]
            # distance (in std units) from each class's typical value
            dist_legit = abs(value - s["legit_mean"]) / s["legit_std"]
            dist_phish = abs(value - s["phish_mean"]) / s["phish_std"]
            direction = "phishing" if dist_phish < dist_legit else "legitimate"
            # how strongly this feature leans, combining global importance
            # with how decisively the value sits on one side
            lean = abs(dist_legit - dist_phish)
            score = importance * lean
            scored.append((name, direction, score))

        scored.sort(key=lambda t: t[2], reverse=True)

        out = []
        for name, direction, score in scored[:top_k]:
            out.append(self._format_entry(name, feature_dict[name], direction, score))
        return out

    def _format_entry(self, name, value, direction, score):
        s = self.stats[name]
        typical = s["phish_mean"] if direction == "phishing" else s["legit_mean"]
        desc = FEATURE_DESCRIPTIONS.get(name, name)

        if direction == "phishing":
            sentence = (
                f"{desc}: this URL's value is {value:g}, which is closer to what "
                f"phishing URLs typically look like (avg {typical:.2f}) than "
                f"legitimate ones."
            )
        else:
            sentence = (
                f"{desc}: this URL's value is {value:g}, which is closer to what "
                f"legitimate URLs typically look like (avg {typical:.2f})."
            )

        return {
            "feature": name,
            "description": desc,
            "value": value,
            "direction": direction,
            "score": float(score),
            "sentence": sentence,
        }


AWARENESS_TIPS = {
    "has_ip": "Legitimate sites almost never use a raw IP address instead of a domain name — treat any link like this as a red flag.",
    "is_shortened": "Shortened links hide the real destination. Hover or use a link-expander before clicking one from an unfamiliar sender.",
    "has_suspicious_tld": "Domain endings like .tk, .xyz, .top or .click are cheap and heavily abused for throwaway phishing sites — be extra cautious.",
    "suspicious_word_count": "Words like 'verify', 'secure', 'urgent' or 'confirm' in a link are classic pressure tactics used to rush you into clicking.",
    "has_https": "No padlock / no HTTPS means the connection isn't encrypted — never enter credentials on such a page.",
    "num_at": "An @ symbol in a URL makes everything before it ignored by the browser — attackers use this to disguise the real destination.",
    "num_hyphens": "Long strings of hyphens in a domain (e.g. paypal-secure-login-update.com) are commonly used to mimic real brand names.",
    "num_subdomains": "Excessive subdomains (e.g. paypal.com.verify-user.ru) try to make a fake domain look like the real one at a glance.",
}


def get_awareness_tips(top_features: list, max_tips: int = 3) -> list:
    tips = []
    for entry in top_features:
        tip = AWARENESS_TIPS.get(entry["feature"])
        if tip and entry["direction"] == "phishing" and tip not in tips:
            tips.append(tip)
        if len(tips) >= max_tips:
            break
    if not tips:
        tips.append(
            "Always double-check the sender and hover over links before clicking, "
            "even when a message looks legitimate."
        )
    return tips
