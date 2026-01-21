
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
import re
import math
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


try:
    from nltk.corpus import stopwords
    RU_STOPWORDS = set(stopwords.words("russian"))
except (ImportError, LookupError):
    RU_STOPWORDS = set()

try:
    import pymorphy2
except ImportError:
    pymorphy2 = None


@lru_cache(maxsize=1)
def get_morph():
    if pymorphy2 is None:
        return None
    return pymorphy2.MorphAnalyzer()


EN_STOPWORDS = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with", "is", "was", "are"}


STOPWORDS = (RU_STOPWORDS | EN_STOPWORDS)


RE_LATIN = re.compile(r"^[a-z][a-z0-9-]*$", re.IGNORECASE)
RE_CYR = re.compile(r"^[а-яё][а-яё-]*$", re.IGNORECASE)



def _require(pkg: str, hint: str) -> None:
    raise RuntimeError(
        f"Не установлен пакет '{pkg}'.\n"
        f"Установка:\n{hint}"
    )


@lru_cache(maxsize=1)
def get_spacy_ru(model: str = "ru_core_news_sm"):

    try:
        import spacy
    except Exception:
        _require("spacy", "pip install spacy\npython -m spacy download ru_core_news_sm")
    try:
        return spacy.load(model)
    except Exception as e:
        raise RuntimeError(
            f"Не удалось загрузить spaCy модель '{model}'. "
            f"Проверь: python -m spacy download {model}\nПричина: {e}"
        )


@lru_cache(maxsize=1)
def get_sentiment_pipeline(model_name: str):

    try:
        from transformers import pipeline  # type: ignore
    except Exception:
        _require("transformers", "pip install transformers torch sentencepiece")
    return pipeline("text-classification", model=model_name, tokenizer=model_name, top_k=None)


@lru_cache(maxsize=1)
def get_translation_pipeline(model_name: str):

    try:
        from transformers import pipeline  # type: ignore
    except Exception:
        _require("transformers", "pip install transformers torch sentencepiece")
    return pipeline("translation", model=model_name, tokenizer=model_name)



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
    author: str
    district: Optional[str]
    date: Optional[str]
    date_iso: str
    season: str
    rating: Optional[float]
    source: Optional[str]

    text_raw: str
    text_clean: str

    tokens_json: str
    lemmas: str
    pos_json: str
    morph_json: str
    deps_json: str
    entities_json: str

    sentiment: Optional[str]
    sentiment_score: Optional[float]

    translation_en: Optional[str]

    topic: Optional[str]
    aspects_json: str
    lemmas_count: int


def _coalesce(obj: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
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
        author = str(_coalesce(obj, ["author", "user", "user_name"], "")).strip() or "unknown"
        review = str(_coalesce(obj, ["review", "text", "comment"], "")).strip()
        district = str(_coalesce(obj, ["district", "area"], "")).strip() or None
        date_val = str(_coalesce(obj, ["date", "created_at", "time"], "")).strip() or None
        source = str(_coalesce(obj, ["source", "platform", "url"], "")).strip() or None
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
            date=date_val,
            rating=rating,
            source=source,
        )

    if suf == ".csv":
        df = pd.read_csv(path)
        return [norm(row.dropna().to_dict()) for _, row in df.iterrows()]

    if suf in (".json",):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [norm(x) for x in obj]
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            return [norm(x) for x in obj["data"]]
        raise ValueError("JSON должен быть списком объектов или {'data':[...]}.")

    if suf in (".jsonl", ".ndjson"):
        out: List[RawReview] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(norm(json.loads(line)))
        return out

    raise ValueError("Unsupported format. Use .csv / .json / .jsonl")



def clean_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u00ad", "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"[\\r\\n\\t]+", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text




RU_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}

def parse_date_ru(s: str):
    if s is None:
        return None
    s = str(s).strip().lower()

    if not s or s in {"nan", "none", "null", "-"}:
        return None

    today = date.today()

    if s == "сегодня":
        return today
    if s == "вчера":
        return today - timedelta(days=1)
    if s == "позавчера":
        return today - timedelta(days=2)

    m = re.search(r"(\d+)\s+дн(я|ей)\s+назад", s)
    if m:
        return today - timedelta(days=int(m.group(1)))

    m = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", s)
    if m:
        d, mo, y = map(int, m.groups())
        return date(y, mo, d)

    m = re.search(r"(\d{1,2})\s+([а-яё]+)(?:\s+(\d{4}))?", s)
    if m:
        d = int(m.group(1))
        month_word = m.group(2)
        y = int(m.group(3)) if m.group(3) else today.year

        mo = RU_MONTHS.get(month_word)
        if not mo:
            return None

        dt = date(y, mo, d)

        if not m.group(3) and dt > today:
            dt = date(y - 1, mo, d)

        return dt

    return None


def season_from_date(dt):
    if dt is None:
        return "Неизвестно"
    m = dt.month
    if m in (12, 1, 2):
        return "Зима"
    if m in (3, 4, 5):
        return "Весна"
    if m in (6, 7, 8):
        return "Лето"
    return "Осень"



def _hash_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"|")
    return h.hexdigest()[:16]


def analyze_docs(
    texts: List[str],
    spacy_model: str,
) -> List[Dict[str, Any]]:

    nlp = get_spacy_ru(spacy_model)

    out: List[Dict[str, Any]] = []
    for doc in nlp.pipe(texts, batch_size=64):
        tokens: List[str] = []
        lemmas: List[str] = []
        pos: List[str] = []
        morph: List[str] = []
        deps: List[Dict[str, Any]] = []

        for t in doc:
            if t.is_space or t.is_punct:
                continue
            tok = (t.text or "").lower().strip()
            if not tok:
                continue

            if not (RE_CYR.match(tok) or RE_LATIN.match(tok) or tok.isdigit()):
                continue

            lem = (t.lemma_ or tok).lower().replace("ё", "е")

            if RE_CYR.match(tok) and (lem == tok):
                mo = get_morph()
                if mo is not None:
                    lem = mo.parse(tok)[0].normal_form.replace("ё", "е")

            if lem in STOPWORDS:
                continue

            tokens.append(tok)
            lemmas.append(lem)
            pos.append(t.pos_)
            morph.append(str(t.morph))

            deps.append({
                "tok": tok,
                "dep": t.dep_,
                "head": int(t.head.i),
                "i": int(t.i),
            })


        PERSON_BAD_LEMMAS = {"комфорт", "кондиционер", "расположить"}
        FORCE_LOC = {"стрелка"}

        entities = []
        for ent in doc.ents:
            toks_ent = [tt for tt in ent if not tt.is_space and not tt.is_punct]
            if not toks_ent:
                continue

            ent_lemmas = [(tt.lemma_ or tt.text).lower().strip().replace("ё", "е") for tt in toks_ent]
            ent_norm = " ".join(ent_lemmas)
            ent_norm = POI_MAP.get(ent_norm, ent_norm)
            has_propn = any(tt.pos_ == "PROPN" for tt in toks_ent)

            label = ent.label_

            if label in {"PERSON", "PER"}:
                if len(toks_ent) == 1 and not has_propn:
                    continue
                if any(l in PERSON_BAD_LEMMAS for l in ent_lemmas):
                    continue
                if all(tt.pos_ in {"VERB", "AUX"} for tt in toks_ent):
                    continue

            if label in {"LOC", "GPE"}:
                if len(toks_ent) == 1 and toks_ent[0].pos_ == "ADJ":
                    continue

            if any(l in FORCE_LOC for l in ent_lemmas):
                label = "LOC"

            entities.append({
                "text": ent.text,
                "norm": ent_norm,
                "label": label,
            })

        out.append({
            "tokens": tokens,
            "lemmas": lemmas,
            "pos": pos,
            "morph": morph,
            "deps": deps,
            "entities": entities,
        })
    return out


def sentiment_batch(texts: List[str], model_name: str) -> List[Tuple[Optional[str], Optional[float]]]:

    clf = get_sentiment_pipeline(model_name)
    res = clf(texts, truncation=True, max_length=256)

    out: List[Tuple[Optional[str], Optional[float]]] = []

    for item in res:
        if isinstance(item, list):
            best = max(item, key=lambda x: x.get("score", 0.0))
        else:
            best = item
        out.append((best.get("label"), float(best.get("score", 0.0))))
    return out


def translate_batch_ru_en(texts: List[str], model_name: str) -> List[str]:
    tr = get_translation_pipeline(model_name)
    out: List[str] = []
    for t in texts:
        t = t[:800]
        try:
            res = tr(t, src_lang="rus_Cyrl", tgt_lang="eng_Latn", max_length=256)
        except TypeError:
            res = tr(t, max_length=256)
        out.append(res[0]["translation_text"])
    return out



POI_MAP = {
    "стрелка": "стрелка (нн)",
    "кремль": "кремль (нн)",
    "нижегородский кремль": "кремль (нн)",
    "рождественская": "ул. рождественская",
}


TOPIC_KEYWORDS = {
    "Чистота": {"чистый", "грязь", "пыль", "уборка", "пятно", "запах"},
    "Сервис": {"персонал", "администратор", "обслуживание", "вежливый", "хамство", "ресепшен"},
    "Расположение": {"центр", "метро", "рядом", "далеко", "транспорт", "вид", "набережная"},
    "Комфорт/шум": {"шум", "тихий", "шумоизоляция", "сосед", "сон", "кровать"},
    "Еда": {"завтрак", "еда", "ресторан", "кофе", "столовая"},
    "Цена": {"цена", "дорого", "дешево", "стоимость", "оплата"},
    "Wi‑Fi/интернет": {"wifi", "вайфай", "интернет"},
}


def classify_topic(lemmas: List[str]) -> Optional[str]:
    if not lemmas:
        return None
    s = set(lemmas)
    best_topic = None
    best_score = 0
    for topic, keys in TOPIC_KEYWORDS.items():
        score = len(s & keys)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic



ASPECTS = {
    "Чистота": {"чистый", "грязь", "пыль", "уборка", "плесень", "таракан", "грязный"},
    "Сервис": {"персонал", "администратор", "обслуживание", "вежливый", "хамство", "ресепшен"},
    "Расположение": {"центр", "метро", "рядом", "далеко", "локация", "транспорт", "набережная", "вид"},
    "Номер": {"номер", "кровать", "ремонт", "шум", "шумоизоляция", "окно", "матрас"},
    "Удобства": {"кондиционер", "wifi", "вайфай", "интернет", "парковка", "лифт", "завтрак"},
    "Цена": {"цена", "дорого", "дешево", "стоимость", "оплата"},
}

def detect_aspects(lemmas: List[str]) -> List[str]:
    if not lemmas:
        return []
    s = set(lemmas)
    hits: List[str] = []
    for aspect, keys in ASPECTS.items():
        if s.intersection(keys):
            hits.append(aspect)
    return hits


def to_typed_reviews(
    raw: List[RawReview],
    spacy_model: str,
    sentiment_model: str,
    translate: bool,
    translation_model: str,
    min_text_len: int = 10,
    enable_sentiment: bool = True,
) -> List[TypedReview]:


    cleaned: List[str] = []
    kept_raw: List[RawReview] = []
    for r in raw:
        text_raw = str(r.review or "").strip()
        if len(text_raw) < min_text_len:
            continue
        text_clean = clean_text(text_raw)
        if len(text_clean) < min_text_len:
            continue
        kept_raw.append(r)
        cleaned.append(text_clean)


    ann = analyze_docs(cleaned, spacy_model=spacy_model)


    sentiments: List[Tuple[Optional[str], Optional[float]]] = [(None, None)] * len(cleaned)
    if enable_sentiment:
        sentiments = sentiment_batch(cleaned, model_name=sentiment_model)


    translations: List[Optional[str]] = [None] * len(cleaned)
    if translate:
        translations = translate_batch_ru_en(cleaned, model_name=translation_model)


    out: List[TypedReview] = []
    for r, text_raw, text_clean, a, (s_label, s_score), tr_en in zip(kept_raw, [x.review for x in kept_raw], cleaned, ann, sentiments, translations):
        d = parse_date_ru(r.date)  # datetime.date | None
        date_iso = d.isoformat() if d else ""
        season = season_from_date(d) if d else "Неизвестно"

        lemmas = a["lemmas"]
        topic = classify_topic(lemmas)
        aspects = detect_aspects(lemmas)

        rid = _hash_id(r.hotel, r.author or "unknown", r.date or "", text_clean)

        out.append(TypedReview(
            review_id=rid,
            hotel=r.hotel,
            author=r.author or "unknown",
            district=r.district,
            date=r.date,
            date_iso=date_iso,
            season=season,
            rating=r.rating,
            source=r.source,

            text_raw=text_raw,
            text_clean=text_clean,

            tokens_json=json.dumps(a["tokens"], ensure_ascii=False),
            lemmas=" ".join(lemmas),
            pos_json=json.dumps(a["pos"], ensure_ascii=False),
            morph_json=json.dumps(a["morph"], ensure_ascii=False),
            deps_json=json.dumps(a["deps"], ensure_ascii=False),
            entities_json=json.dumps(a["entities"], ensure_ascii=False),

            sentiment=s_label,
            sentiment_score=s_score,

            translation_en=tr_en,

            topic=topic,
            aspects_json=json.dumps(aspects, ensure_ascii=False),
            lemmas_count=len(lemmas),
        ))

    return out


def typed_reviews_df(typed: Iterable[TypedReview]) -> pd.DataFrame:
    return pd.DataFrame([t.__dict__ for t in typed])


def build_lemma_counts(typed: Iterable[TypedReview]) -> pd.DataFrame:

    rows: List[Dict[str, Any]] = []
    for t in typed:
        lems = [x for x in (t.lemmas or "").split() if x]
        c = Counter(lems)
        for lemma, cnt in c.items():
            rows.append({
                "hotel": t.hotel,
                "district": t.district,
                "season": t.season,
                "sentiment": t.sentiment,
                "lemma": lemma,
                "count": int(cnt),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby(["hotel", "district", "season", "sentiment", "lemma"], as_index=False)["count"].sum()



def build_topics(typed: Iterable[TypedReview]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in typed:
        if not t.topic:
            continue
        rows.append({
            "hotel": t.hotel,
            "district": t.district,
            "season": t.season,
            "sentiment": t.sentiment,
            "topic": t.topic,
            "reviews_count": 1
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby(["hotel", "district", "season", "sentiment", "topic"], as_index=False)["reviews_count"].sum()


def build_entities(typed: Iterable[TypedReview]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in typed:
        try:
            ents = json.loads(t.entities_json or "[]")
        except Exception:
            ents = []
        for e in ents:
            txt = str(e.get("norm") or e.get("text", "")).strip()
            lab = str(e.get("label", "")).strip()
            if not txt or not lab:
                continue
            rows.append({
                "review_id": t.review_id,
                "hotel": t.hotel,
                "district": t.district,
                "season": t.season,
                "sentiment": t.sentiment,
                "entity_text": txt,
                "entity_label": lab,
                "count": 1,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby(["hotel", "district", "season", "sentiment", "entity_label", "entity_text"], as_index=False)["count"].sum()




def build_aspect_sentiment(typed: Iterable[TypedReview]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in typed:
        try:
            aspects = json.loads(t.aspects_json or "[]")
        except Exception:
            aspects = []
        if not aspects:
            continue

        for a in aspects:
            a = str(a).strip()
            if not a:
                continue
            rows.append({
                "hotel": t.hotel,
                "district": t.district,
                "season": t.season,
                "sentiment": t.sentiment,
                "aspect": a,
                "count": 1,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.groupby(
        ["hotel", "district", "season", "sentiment", "aspect"],
        as_index=False
    )["count"].sum()


def build_complaints(
    typed: Iterable[TypedReview],
    min_count: int = 2,
    spacy_model: str = "ru_core_news_sm",
) -> pd.DataFrame:

    nlp = get_spacy_ru(spacy_model)
    mo = get_morph()

    COMPLAINT_ADVS = {"плохо", "ужасно", "слишком", "чересчур", "сильно"}

    def norm_lemma(tok) -> Optional[str]:
        if tok.is_space or tok.is_punct:
            return None
        t = (tok.text or "").lower().strip()
        if not t:
            return None
        if t.isdigit():
            return None
        if not (RE_CYR.match(t) or RE_LATIN.match(t)):
            return None

        lem = (tok.lemma_ or t).lower().replace("ё", "е")
        if RE_CYR.match(t) and (lem == t) and mo is not None:
            try:
                lem = mo.parse(t)[0].normal_form.replace("ё", "е")
            except Exception:
                pass
        return lem

    neg_items: List[TypedReview] = []
    texts: List[str] = []
    for t in typed:
        if (t.sentiment or "").upper() != "NEGATIVE":
            continue
        if not (t.text_clean or "").strip():
            continue
        neg_items.append(t)
        texts.append(t.text_clean)

    rows: List[Dict[str, Any]] = []

    for t, doc in zip(neg_items, nlp.pipe(texts, batch_size=32)):

        uni: Counter = Counter()
        for tok in doc:
            lem = norm_lemma(tok)
            if not lem:
                continue


            if lem in STOPWORDS:
                continue

            uni[lem] += 1

        for term, cnt in uni.items():
            rows.append({
                "hotel": t.hotel,
                "district": t.district,
                "season": t.season,
                "term": term,
                "n": 1,
                "count": int(cnt),
            })

        bi: Counter = Counter()

        for tok in doc:
            if tok.dep_ == "neg":
                head = tok.head
                hlem = norm_lemma(head)
                if hlem and (hlem not in STOPWORDS):
                    bi[f"{tok.lower_} {hlem}"] += 1

            if tok.dep_ == "amod" and tok.pos_ == "ADJ":
                head = tok.head
                if head.pos_ in {"NOUN", "PROPN"}:
                    a = norm_lemma(tok)
                    n = norm_lemma(head)
                    if a and n and (a not in STOPWORDS) and (n not in STOPWORDS):
                        bi[f"{a} {n}"] += 1

            if tok.pos_ in {"NOUN", "PROPN"} and tok.head is not None:
                head = tok.head
                if head.pos_ in {"NOUN", "PROPN"} and tok.dep_ in {"nmod", "compound", "appos"}:
                    h = norm_lemma(head)
                    n = norm_lemma(tok)
                    if h and n and (h not in STOPWORDS) and (n not in STOPWORDS):
                        bi[f"{h} {n}"] += 1

            if tok.dep_ == "advmod" and tok.pos_ == "ADV":
                a = norm_lemma(tok)
                if a and a in COMPLAINT_ADVS:
                    head = tok.head
                    h = norm_lemma(head)
                    if h and (h not in STOPWORDS):
                        bi[f"{a} {h}"] += 1

        for term, cnt in bi.items():
            rows.append({
                "hotel": t.hotel,
                "district": t.district,
                "season": t.season,
                "term": term,
                "n": 2,
                "count": int(cnt),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.groupby(["hotel", "district", "season", "n", "term"], as_index=False)["count"].sum()
    df = df[df["count"] >= int(min_count)].sort_values("count", ascending=False)
    return df


def build_tfidf_by_group(
    typed: Iterable[TypedReview],
    group_field: str,
    top_k: int = 50,
    min_df: int = 2,
) -> pd.DataFrame:

    group_tf: Dict[str, Counter] = {}
    group_len: Dict[str, int] = {}

    for t in typed:
        g = getattr(t, group_field, None) or "unknown"
        lems = [x for x in (t.lemmas or "").split() if x and x not in STOPWORDS]
        if not lems:
            continue
        c = group_tf.get(g)
        if c is None:
            c = Counter()
            group_tf[g] = c
            group_len[g] = 0
        c.update(lems)
        group_len[g] += len(lems)

    groups = list(group_tf.keys())
    N = len(groups)
    if N == 0:
        return pd.DataFrame([])

    df = Counter()
    for g in groups:
        df.update(group_tf[g].keys())

    rows: List[Dict[str, Any]] = []
    for g in groups:
        total = max(1, group_len.get(g, 1))

        scores = {}
        for term, tf in group_tf[g].items():
            if df[term] < min_df:
                continue
            tf_norm = tf / total
            idf = (math.log((N + 1) / (df[term] + 1)) + 1.0)
            scores[term] = tf_norm * idf
        for term, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
            rows.append({
                "group_field": group_field,
                "group": g,
                "term": term,
                "tfidf": float(score),
                "tf": int(group_tf[g][term]),
                "df": int(df[term]),
                "total_terms": int(total),
            })
    return pd.DataFrame(rows)


def export_outputs(typed: Iterable[TypedReview], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    df_reviews = typed_reviews_df(typed)
    df_counts = build_lemma_counts(typed)
    df_topics = build_topics(typed)
    df_ents = build_entities(typed)
    df_aspects = build_aspect_sentiment(typed)
    df_complaints = build_complaints(typed, min_count=2)
    df_tfidf_hotel = build_tfidf_by_group(typed, group_field="hotel")
    df_tfidf_district = build_tfidf_by_group(typed, group_field="district")
    df_tfidf_season = build_tfidf_by_group(typed, group_field="season")

    paths: Dict[str, Path] = {}

    p_reviews = out_dir / "reviews_typed.csv"
    df_reviews.to_csv(p_reviews, index=False)
    paths["reviews_typed"] = p_reviews

    p_counts = out_dir / "word_stats.csv"
    df_counts.to_csv(p_counts, index=False)
    paths["word_stats"] = p_counts

    p_topics = out_dir / "topics.csv"
    df_topics.to_csv(p_topics, index=False)
    paths["topics"] = p_topics

    p_ents = out_dir / "entities.csv"
    df_ents.to_csv(p_ents, index=False)
    paths["entities"] = p_ents

    p_aspects = out_dir / "aspect_sentiment.csv"
    df_aspects.to_csv(p_aspects, index=False)
    paths["aspect_sentiment"] = p_aspects

    p_compl = out_dir / "complaints.csv"
    df_complaints.to_csv(p_compl, index=False)
    paths["complaints"] = p_compl

    p_tfidf_h = out_dir / "tfidf_hotel.csv"
    df_tfidf_hotel.to_csv(p_tfidf_h, index=False)
    paths["tfidf_hotel"] = p_tfidf_h

    p_tfidf_d = out_dir / "tfidf_district.csv"
    df_tfidf_district.to_csv(p_tfidf_d, index=False)
    paths["tfidf_district"] = p_tfidf_d

    p_tfidf_s = out_dir / "tfidf_season.csv"
    df_tfidf_season.to_csv(p_tfidf_s, index=False)
    paths["tfidf_season"] = p_tfidf_s

    return paths



def main() -> None:
    ap = argparse.ArgumentParser(
        description="NLP обработка отзывов: токенизация/леммы/POS/NER, тональность, перевод, агрегации под визуализации."
    )
    ap.add_argument("--input", required=True, help="Путь к .csv/.json/.jsonl с отзывами")
    ap.add_argument("--output", default="out", help="Папка для результатов")
    ap.add_argument("--min_text_len", type=int, default=10, help="Минимальная длина отзыва (после очистки)")
    ap.add_argument("--spacy_model", default="ru_core_news_sm", help="spaCy RU модель (например ru_core_news_sm)")
    ap.add_argument("--sentiment_model", default="blanchefort/rubert-base-cased-sentiment-rusentiment",
                    help="HF модель тональности (NEGATIVE/POSITIVE/NEUTRAL)")
    ap.add_argument("--no_sentiment", action="store_true", help="Не считать тональность (ускорить)")
    ap.add_argument("--translate", action="store_true", help="Добавить перевод RU->EN (медленно)")
    ap.add_argument("--translation_model", default="facebook/nllb-200-distilled-600M",
                    help="HF модель перевода (например facebook/nllb-200-distilled-600M)")

    args = ap.parse_args()

    raw = load_reviews(Path(args.input))
    typed = to_typed_reviews(
        raw,
        spacy_model=args.spacy_model,
        sentiment_model=args.sentiment_model,
        translate=args.translate,
        translation_model=args.translation_model,
        min_text_len=args.min_text_len,
        enable_sentiment=not args.no_sentiment,
    )

    paths = export_outputs(typed, Path(args.output))
    print(f"OK. Typed reviews: {len(typed)}")
    for k, p in paths.items():
        print(f"Saved {k}: {p}")


if __name__ == "__main__":
    main()