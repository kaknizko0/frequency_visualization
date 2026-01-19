from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pymorphy2
from razdel import tokenize as razdel_tokenize



try:
    from nltk.corpus import stopwords

    RU_STOPWORDS = set(stopwords.words("russian"))
except (ImportError, LookupError):
    RU_STOPWORDS = set()

DOMAIN_WORDS = {
    "отель",
    "гостиница",
    "номер",
    "заселение",
    "выселение",
    "очень",
    "вообще",
    "просто",
    "это",
    "всё",
    "все",
    "который",
}

EN_STOPWORDS = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with", "is", "was", "are"}
STOPWORDS = RU_STOPWORDS | DOMAIN_WORDS | EN_STOPWORDS


RE_LATIN = re.compile(r"^[a-z][a-z0-9-]*$", re.IGNORECASE)
RE_CYR = re.compile(r"^[а-яё][а-яё-]*$", re.IGNORECASE)

morph = pymorphy2.MorphAnalyzer()


TOPIC_LEMMAS: dict[str, set[str]] = {
    "Чистота": {"чистый", "чистота", "грязь", "грязный", "уборка", "пыль", "мусор", "плесень", "запах", "вонь"},
    "Сервис": {"персонал", "администратор", "обслуживание", "сервис", "вежливый", "хамить", "ресепшен", "встречать", "помогать"},
    "Номер": {"кровать", "матрас", "подушка", "ремонт", "шум", "тихо", "шумоизоляция", "окно", "кондиционер", "полотенце"},
    "Локация": {"центр", "метро", "рядом", "далеко", "расположение", "транспорт", "улица", "район"},
    "Еда": {"завтрак", "еда", "ресторан", "кафе", "шведский", "стол", "вкусный"},
    "Цена": {"цена", "стоимость", "дорого", "дешево", "переплата", "оплата"},
}


@dataclass(frozen=True)
class RawReview:
    hotel: str
    review: str
    author: str = "unknown"
    district: Optional[str] = None
    date: Optional[str] = None
    rating: Optional[float] = None
    source: Optional[str] = None


@dataclass(frozen=True)
class TypedReview:
    review_id: str
    hotel: str
    district: Optional[str]
    author: str
    date: Optional[str]
    season: str
    rating: Optional[float]
    sentiment: str
    source: Optional[str]
    text_raw: str
    text_clean: str
    lemmas: str
    lemmas_count: int
    topic: str


def _coalesce(obj: Dict[str, Any], keys: List[str] | str, default: Any = "") -> Any:
    if isinstance(keys, str):
        keys = [keys]
    for k in keys:
        if k in obj and obj[k] is not None and str(obj[k]).strip() != "":
            return obj[k]
    return default


def load_reviews(path: Path) -> List[RawReview]:
    suf = path.suffix.lower()

    def norm(obj: Dict[str, Any]) -> RawReview:
        hotel = str(_coalesce(obj, ["hotel", "hotel_name", "name"], "")).strip()
        review = str(_coalesce(obj, ["review", "text", "comment"], "")).strip()
        author = str(_coalesce(obj, ["author", "user", "user_name"], "")).strip() or "unknown"
        district = str(_coalesce(obj, ["district", "area", "region"], "")).strip() or None
        date_s = str(_coalesce(obj, ["date", "created_at", "time"], "")).strip() or None
        source = str(_coalesce(obj, ["source", "platform"], "")).strip() or None

        rating_raw = _coalesce(obj, ["rating", "stars", "score"], None)
        rating: Optional[float]
        try:
            rating = float(rating_raw) if rating_raw is not None and str(rating_raw).strip() != "" else None
        except (TypeError, ValueError):
            rating = None

        return RawReview(
            hotel=hotel,
            review=review,
            author=author,
            district=district,
            date=date_s,
            rating=rating,
            source=source,
        )

    if suf == ".csv":
        df = pd.read_csv(path)
        return [norm(r) for r in df.to_dict(orient="records")]

    if suf == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a list, but got {type(data)}")
        return [norm(x) for x in data]

    if suf == ".jsonl":
        out: List[RawReview] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(norm(json.loads(line)))
        return out

    raise ValueError("Unsupported format. Use .csv / .json / .jsonl")


def clean_reviews(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u00ad", "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return [t.text.lower() for t in razdel_tokenize(text)]


@lru_cache(maxsize=200_000)
def _analyze(word: str) -> Tuple[str, Optional[str]]:
    p = morph.parse(word)[0]
    return p.normal_form, p.tag.POS


def lemmatize(tokens: List[str], keep_pos: Optional[set] = None) -> List[str]:
    if keep_pos is None:
        keep_pos = {"NOUN", "ADJF", "ADVB"}

    lemmas: List[str] = []
    for t in tokens:
        t = (t or "").strip().lower()
        if not t:
            continue

        t = t.replace("ё", "е").replace("wi-fi", "wifi").replace("wi_fi", "wifi")

        if RE_LATIN.match(t):
            if t in STOPWORDS:
                continue
            if len(t) <= 2 and t not in {"tv", "spa"}:
                continue
            lemmas.append(t)
            continue

        if not RE_CYR.match(t):
            continue

        if len(t) <= 3:
            continue

        lemma, pos = _analyze(t)
        if lemma in STOPWORDS:
            continue
        if pos is None or pos not in keep_pos:
            continue
        lemmas.append(lemma)

    return lemmas


_RU_MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}


def _season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "Зима"
    if m in (3, 4, 5):
        return "Весна"
    if m in (6, 7, 8):
        return "Лето"
    if m in (9, 10, 11):
        return "Осень"
    return "Неизвестно"


def parse_date_any(raw: Optional[str]) -> Optional[str]:

    if not raw:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None

    today = date.today()

    if s == "сегодня":
        return today.isoformat()
    if s == "вчера":
        return (today - timedelta(days=1)).isoformat()

    m = re.search(r"(\d+)\s+дн", s)
    if m and "назад" in s:
        return (today - timedelta(days=int(m.group(1)))).isoformat()

    m = re.search(r"(\d{1,2})\s+([а-я]+)\s+(\d{4})", s)
    if m:
        d = int(m.group(1))
        month = _RU_MONTHS.get(m.group(2))
        y = int(m.group(3))
        if month:
            try:
                return date(y, month, d).isoformat()
            except ValueError:
                pass

    try:
        ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.isna(ts):
            return None
        return ts.date().isoformat()
    except Exception:
        return None


def sentiment_from_rating(r: Optional[float]) -> str:
    if r is None:
        return "unknown"
    if r >= 4:
        return "positive"
    if r <= 2:
        return "negative"
    return "neutral"


def topic_from_lemmas(lemmas: List[str]) -> str:
    if not lemmas:
        return "Другое"

    counts: dict[str, int] = {}
    s = set(lemmas)
    for topic, vocab in TOPIC_LEMMAS.items():
        counts[topic] = len(s & vocab)

    best_topic, best_score = max(counts.items(), key=lambda x: x[1])
    return best_topic if best_score > 0 else "Другое"


def stable_review_id(hotel: str, author: str, text_clean: str) -> str:
    s = f"{hotel.lower()}||{author.lower()}||{text_clean.lower()}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def to_typed_reviews(raw: Iterable[RawReview], min_text_len: int = 10) -> List[TypedReview]:
    out: List[TypedReview] = []

    for r in raw:
        if not r.hotel or not r.review:
            continue

        text_raw = str(r.review)
        text_clean = clean_reviews(text_raw)
        if len(text_clean) < min_text_len:
            continue

        toks = tokenize(text_clean)
        lems = lemmatize(toks)

        date_iso = parse_date_any(r.date)
        season = _season_from_month(int(date_iso[5:7])) if date_iso else "Неизвестно"

        sentiment = sentiment_from_rating(r.rating)
        topic = topic_from_lemmas(lems)

        rid = stable_review_id(r.hotel, r.author, text_clean)

        out.append(
            TypedReview(
                review_id=rid,
                hotel=r.hotel,
                district=r.district,
                author=r.author or "unknown",
                date=date_iso,
                season=season,
                rating=r.rating,
                sentiment=sentiment,
                source=r.source,
                text_raw=text_raw,
                text_clean=text_clean,
                lemmas=" ".join(lems),
                lemmas_count=len(lems),
                topic=topic,
            )
        )

    return out


def typed_reviews_df(typed: Iterable[TypedReview]) -> pd.DataFrame:
    return pd.DataFrame([asdict(x) for x in typed])


def build_lemma_counts_by_hotel(typed: Iterable[TypedReview]) -> pd.DataFrame:
    agg: Dict[str, Counter[str]] = defaultdict(Counter)
    for r in typed:
        if r.lemmas:
            agg[r.hotel].update(r.lemmas.split())

    rows: list[dict[str, Any]] = []
    for hotel, counts in agg.items():
        for lemma, cnt in counts.items():
            rows.append({"hotel": hotel, "lemma": lemma, "count": int(cnt)})

    df = pd.DataFrame(rows, columns=["hotel", "lemma", "count"])
    if not df.empty:
        df = df.sort_values(["hotel", "count"], ascending=[True, False]).reset_index(drop=True)
    return df


def build_word_stats(typed: Iterable[TypedReview]) -> pd.DataFrame:
    counter: dict[tuple[str, str, str], int] = defaultdict(int)

    for r in typed:
        if not r.lemmas:
            continue
        for lemma in r.lemmas.split():
            counter[(r.season, r.sentiment, lemma)] += 1

    rows = [
        {"season": k[0], "sentiment": k[1], "lemma": k[2], "count": v}
        for k, v in counter.items()
    ]

    df = pd.DataFrame(rows, columns=["season", "sentiment", "lemma", "count"])
    if not df.empty:
        df = df.sort_values(["season", "sentiment", "count"], ascending=[True, True, False]).reset_index(drop=True)
    return df


def build_topics_stats(typed: Iterable[TypedReview]) -> pd.DataFrame:
    counter: dict[tuple[str, str, str], int] = defaultdict(int)
    for r in typed:
        counter[(r.season, r.sentiment, r.topic)] += 1

    rows = [
        {"season": k[0], "sentiment": k[1], "topic": k[2], "reviews_count": v}
        for k, v in counter.items()
    ]

    df = pd.DataFrame(rows, columns=["season", "sentiment", "topic", "reviews_count"])
    if not df.empty:
        df = df.sort_values(["season", "sentiment", "reviews_count"], ascending=[True, True, False]).reset_index(drop=True)
    return df


def export_outputs(typed: List[TypedReview], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    df_reviews = typed_reviews_df(typed)
    df_counts = build_lemma_counts_by_hotel(typed)
    df_word_stats = build_word_stats(typed)
    df_topics = build_topics_stats(typed)

    paths: dict[str, Path] = {
        "reviews_typed": out_dir / "reviews_typed.csv",
        "counts_typed": out_dir / "counts_typed.csv",
        "word_stats": out_dir / "word_stats.csv",
        "topics": out_dir / "topics.csv",
    }

    df_reviews.to_csv(paths["reviews_typed"], index=False)
    df_counts.to_csv(paths["counts_typed"], index=False)
    df_word_stats.to_csv(paths["word_stats"], index=False)
    df_topics.to_csv(paths["topics"], index=False)

    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="NLP обработка отзывов: очистка, леммы, сезоны/сентимент/темы.")
    ap.add_argument("--input", required=True, help="Путь к .csv/.json/.jsonl с отзывами")
    ap.add_argument("--output", default="out", help="Папка для результатов")
    ap.add_argument("--min-text-len", type=int, default=10, help="Минимальная длина текста после очистки")
    args = ap.parse_args()

    raw = load_reviews(Path(args.input))
    typed = to_typed_reviews(raw, min_text_len=args.min_text_len)
    paths = export_outputs(typed, Path(args.output))

    print(f"OK. Typed reviews: {len(typed)}")
    for k, p in paths.items():
        print(f"Saved {k}: {p}")


if __name__ == "__main__":
    main()
