"""Phase 2 preprocessing pipeline.

Order is fixed to match ARCHITECTURE.md §2.2:
1. noise removal
2. mixed-script tokenization
3. script validation
4. bilingual stopword removal
5. lemmatization/stemming

This is a rule-based preprocessing layer intended for the fixed 6,583 filtered
corpus from Phase 1. It performs no silent script guessing and flags ambiguous
or garbage tokens rather than converting them.
"""
import html
import re
from typing import Dict, List


MARKERS_PATH = r"C:\Users\User\Desktop\NLP\data\roman_hindi_markers.json"

ROMAN_HINDI_MARKERS = [
    "hai", "hain", "kya", "nahi", "bhai", "yaar", "acha", "achha", "kar",
    "karo", "ka", "ke", "ki", "tu", "tum", "main", "mein", "kaun", "kaha",
    "kahan", "kuch", "bhi", "raha", "rahi", "gaya", "gayi", "log", "sab"
]

EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "to",
    "of", "in", "on", "at", "by", "with", "as", "is", "was", "are", "were",
    "be", "been", "being", "it", "its", "this", "that", "these", "those",
    "he", "she", "they", "we", "you", "i", "me", "my", "your", "our", "us",
    "do", "does", "did", "have", "has", "had", "yes", "from", "into", "out",
    "up", "down", "so", "too", "very", "just", "can", "could", "would",
    "should", "will", "about", "after", "before", "under", "over", "again",
    "there", "here", "all", "some", "any", "many", "much", "more", "most",
    "such", "when", "than", "through", "during", "which", "while", "because"
}

HI_STOPWORDS = {
    "hai", "hain", "nahin", "bhi", "sab", "log", "ka", "ki", "ke", "se",
    "mein", "main", "tum", "tu", "apni", "ham", "hum", "aap", "aur", "par",
    "ek", "do", "thi", "raha", "rahi", "gaya", "gayi", "kar", "karo",
    "achha", "acha", "yaar", "bhai", "kuch", "kaha"
}

NEGATION_AND_QUESTION_WORDS = {
    "nahi", "kya", "kahan", "kaun", "kyun", "kaise",
    "not", "no", "what", "where", "who", "why", "how"
}

ROMAN_HI_STOPWORDS = set(HI_STOPWORDS)

LATIN_RE = re.compile(r"[A-Za-z]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HTML_TAG_RE = re.compile(r"<.*?>")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAD6\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u27BF\U0001F900-\U0001F9FF\uFE0F]")
REPEATED_CHAR_RE = re.compile(r"(\w)\1{2,}")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = spacy.blank("en")


def export_roman_hindi_markers(path: str = MARKERS_PATH) -> None:
    import json
    payload = {"roman_hindi_markers": sorted(set(ROMAN_HINDI_MARKERS))}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def noise_removal(text: str) -> str:
    if text is None:
        return ""
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = re.sub(r"[\u200d\u00a0]", " ", text)
    text = re.sub(r"[#*_~`>\-]+", " ", text)
    text = re.sub(r"[!?.,;:/\\|()\[\]{}]+", " ", text)
    text = REPEATED_CHAR_RE.sub(r"\1\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def mixed_script_tokenize(text: str) -> List[str]:
    cleaned = noise_removal(text)
    if not cleaned:
        return []
    tokens = re.findall(
        r"[\u0900-\u097F]+(?:[\u200c\u200d][\u0900-\u097F]+)*|[A-Za-z]+(?:'[A-Za-z]+)?|[A-Za-z0-9]+",
        cleaned,
    )
    return [t for t in tokens if t.strip()]


def classify_script_token(token: str) -> str:
    if not token or not token.strip():
        return "garbage"
    contains_latin = bool(LATIN_RE.search(token))
    contains_devanagari = bool(DEVANAGARI_RE.search(token))
    contains_digit = any(ch.isdigit() for ch in token)
    letters_only = token.isalpha()

    if contains_digit:
        return "garbage"
    if contains_latin and contains_devanagari:
        return "mixed"
    if contains_latin and not contains_devanagari:
        return "latin"
    if contains_devanagari and not contains_latin:
        return "devanagari"
    if letters_only:
        return "garbage"
    return "garbage"


def validate_script_tags(tokens: List[str]) -> List[Dict[str, str]]:
    tagged = []
    for token in tokens:
        script = classify_script_token(token)
        tagged.append({"token": token, "script": script})
    return tagged


def remove_stopwords(tokens: List[str], tags: List[str]) -> List[str]:
    kept: List[str] = []
    for token, tag in zip(tokens, tags):
        normalized = token.lower()
        if normalized in NEGATION_AND_QUESTION_WORDS:
            kept.append(token)
            continue
        if tag == "latin":
            if normalized in EN_STOPWORDS:
                continue
            if normalized in ROMAN_HI_STOPWORDS:
                continue
        if tag in {"devanagari", "mixed"}:
            if normalized in HI_STOPWORDS:
                continue
        if tag == "garbage":
            continue
        kept.append(token)
    return kept


def _hindi_suffix_stem(token: str) -> str:
    suffixes = ["wala", "wali", "waale", "kar", "ke", "ki", "se", "me", "mai", "mein"]
    lowered = token.lower()
    for suffix in suffixes:
        if lowered.endswith(suffix) and len(lowered) > len(suffix) + 2:
            return lowered[: -len(suffix)]
    return lowered


def lemmatize_or_stem(tokens: List[str], tags: List[str]) -> List[str]:
    results: List[str] = []
    for token, tag in zip(tokens, tags):
        if tag == "latin":
            doc = nlp(token)
            lemma = doc[0].lemma_.lower() if len(doc) else token.lower()
            results.append(lemma if lemma and lemma != "-PRON-" else token.lower())
        elif tag == "devanagari":
            results.append(_hindi_suffix_stem(token))
        else:
            results.append(token)
    return results


def preprocess_text(text: str, track: str = "term_work") -> Dict[str, object]:
    cleaned = noise_removal(text)
    tokens = mixed_script_tokenize(cleaned)
    tagged = validate_script_tags(tokens)

    if track == "model_input":
        return {
            "raw": text,
            "cleaned": cleaned,
            "tokens": tokens,
            "tagged": tagged,
        }

    scripts = [entry["script"] for entry in tagged]
    valid_pairs = [(token, script) for token, script in zip(tokens, scripts) if script != "garbage"]
    kept_tokens = [token for token, _ in valid_pairs]
    kept_scripts = [script for _, script in valid_pairs]
    filtered_pairs = [
        (token, script)
        for token, script in zip(kept_tokens, kept_scripts)
        if not ((script == "latin") and (token.lower() in EN_STOPWORDS or token.lower() in ROMAN_HI_STOPWORDS))
        and not ((script in {"devanagari", "mixed"}) and (token.lower() in HI_STOPWORDS))
    ]
    filtered_tokens = [token for token, _ in filtered_pairs]
    filtered_scripts = [script for _, script in filtered_pairs]
    lemmatized = lemmatize_or_stem(filtered_tokens, filtered_scripts)
    return {
        "raw": text,
        "cleaned": cleaned,
        "tokens": tokens,
        "tagged": tagged,
        "filtered_tokens": filtered_tokens,
        "lemmatized": lemmatized,
    }


def pipeline_stub(text: str) -> Dict:
    return preprocess_text(text)


if __name__ == "__main__":
    export_roman_hindi_markers(MARKERS_PATH)
    sample = "kya bhai, ye class mein aur demo hai!!! https://x.com @user 😄"
    print(pipeline_stub(sample))
