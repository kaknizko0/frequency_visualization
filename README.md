# Hotel Reviews Pipeline: Scraping (Yandex Maps) + NLP (spaCy/Transformers)



> исполняемые модули:
> - `hotel_scraper.py` — собирает карточки отелей (title/link/district)
> - `reviews_scraper.py` — собирает отзывы по карточке отеля
> - `main.py` — оркестратор: отели → отзывы → `output/reviews_raw.csv`
> - `nlp_hotels.py` — NLP-пайплайн и генерация таблиц в `out/`

---

## 1) End-to-end поток данных

### 1.1 сбор отелей
**вход:** список районов (в коде `hotel_scraper.py:DISTRICTS`)  
**выход:** `output/hotels.json`:
```json
[
  {"title": "…", "link": "https://yandex.ru/maps/org/.../", "district": "…", "city": "Нижний Новгород"},
  ...
]
```

### 1.2 сбор отзывов
**вход:** `output/hotels.json`  
**выход:** `output/reviews_raw.csv` (1 строка = 1 отзыв)

### 1.3 NLP и таблицы
**вход:** `output/reviews_raw.csv`  
**выход:** `out/*.csv` (см. раздел 7)

---

## 2) установка и зависимости (кратко)

```bash
python -m venv venv
source venv/bin/activate

pip install pandas numpy
pip install playwright
pip install spacy pymorphy2 pymorphy2-dicts-ru
pip install transformers torch sentencepiece

playwright install chromium
python -m spacy download ru_core_news_sm
```

(опционально) NLTK stopwords:
```bash
pip install nltk
python -c "import nltk; nltk.download('stopwords')"
```

---

## 3) скрапинг: 

### 3.1 почему Playwright, а не requests/bs4
Yandex Maps — динамический SPA:
- контент подгружается при скролле
- часть данных появляется после JS
- периодически появляется проверка “Вы человек”/капча
пПоэтому используется **Playwright + Chromium** (реальный браузер), а не статический HTML-парсинг

### 3.2 Persistent Browser Context (профиль браузера)
`hotel_scraper.py` и `reviews_scraper.py` запускают браузер через:
`chromium.launch_persistent_context(user_data_dir=...)`
зачем это сделано:
- куки/локал-сторедж/сессии сохраняются
- меньше “повторных” баннеров
- снижает частоту капчи

папки профилей:
- `output/pw_profile_maps` — поиск/карточки
- `output/pw_profile_reviews` — отзывы

### 3.3 Headless выключен
по умолчанию `headless=False`, потому что:
- капчу приходится решать руками
- в headless капча чаще/жестче

### 3.4 детект и обработка капчи
капча/проверка определяется через:
- URL содержит `showcaptcha` или `checkcaptcha`
- на странице есть текст `Вы человек`
- присутствует `iframe` с `src*='captcha'`

если капча обнаружена:
- скрипт печатает инструкцию
- ждёт подтверждение в терминале (`input(...)`)
- после этого продолжает

---

## 4) `hotel_scraper.py` — сбор списка отелей (карточек)

### 4.1 география и query-URL
в коде фиксированы:
- `CITY_CODE = 47`
- `CITY_SLUG = "nizhny-novgorod"`
- `DISTRICTS = [...]` (районы)

URL для поиска строится так:
- базовый путь: `https://yandex.ru/maps/{CITY_CODE}/{CITY_SLUG}/search/`
- запрос: `"гостиницы {district} район"` (для каждого района)
- query кодируется через `urllib.parse.quote`

функция: `get_url(district: Optional[str]) -> str`

### 4.2 навигация и таймауты
в `scrape_async()`:
- создаётся persistent context (locale=`ru-RU`, args `--no-sandbox`, `--disable-dev-shm-usage`)
- создаётся `page`
- `page.set_default_timeout(timeout_ms)` (по умолчанию 60 сек)
- `page.goto(url, wait_until="domcontentloaded")`
- короткая пауза `wait_for_timeout(500)`

### 4.3 скролл списка результатов
функция: `_scroll_results(page)`

алгоритм (детерминированный):
- на каждой итерации считается количество карточек результатов:
  - пробует `query_selector_all("li[class*...search-business-snippet-view, li.search-business-snippet-view")`
  - если селектор не сработал — пусто
- если количество карточек растёт → продолжаем
- если не растёт → увеличивается `stagnation`
- остановка при `max_stagnation = 4` (4 “пустых” раунда)

скролл делается так:
- пытается найти контейнер `.scroll__content` и навести курсор (чтобы wheel влиял на нужную область)
- `page.mouse.wheel(0, 12000)`
- пауза `wait_for_timeout(700)`

### 4.4 извлечение карточек отелей
функция: `_extract_hotels_from_page(page, district)`

базовый селектор карточки:
- `.search-business-snippet-view`

извлечение названия:
- последовательно пробуются:
  - `[class*='search-business-snippet-view__title']`
  - `[class*='org-title']`
  - `[class*='title']`

извлечение ссылки:
- последовательно пробуются:
  - `a[href*='/org/']`
  - `a[class*='link']`

нормализация ссылки:
- `normalize_org_url()`:
  - добавляет `https://yandex.ru` к относительным ссылкам
  - убирает query (`?...`)
  - срезает хвостовые страницы типа `/gallery`, `/reviews`, `/photos` и т.п.
  - приводит к каноническому виду с завершающим `/`

дедупликация:
- в `scrape_async()` ведётся `seen: set[str]` по `link`, чтобы убрать дубликаты по районам/страницам

### 4.5 Sync wrapper
функция `scrape()` делает удобный sync API поверх async:
- если event loop не запущен → `asyncio.run(...)`
- если loop уже есть (например, Jupyter) → запускает `asyncio.run` внутри `ThreadPoolExecutor`

---

## 5) `reviews_scraper.py` — сбор отзывов

### 5.1 нормализация URL на страницу отзывов
функция: `to_reviews_url(url)`

логика:
- отрезает query (`?...`)
- если URL заканчивается на `/gallery` → заменяет на `/reviews/`
- если URL не заканчивается на `/reviews/` → добавляет `reviews/`

это важно, потому что карточки могут открываться в разных “режимах” (галерея/главная), а отзывы нужно всегда тянуть со стабильного `/reviews/`

### 5.2 инициализация браузера
класс: `AsyncReviewsScraper`

`init_playwright()`:
- `chromium.launch_persistent_context(...)` с:
  - `user_data_dir="output/pw_profile_reviews"`
  - `locale="ru-RU"`
  - args: `["--no-sandbox", "--disable-dev-shm-usage"]`
- создаёт `page` и ставит default timeout (60 сек)

### 5.3 “кнопки согласия” и баннеры
метод `_try_click_common_buttons()` кликает типовые кнопки:
- `Принять`, `Согласен`, `ОК`, `Понятно`, `Закрыть`

это делается:
- сразу после `goto`
- после каждой прокрутки (на случай всплывающих диалогов)

### 5.4 обработка капчи
методы:
- `_looks_like_captcha()` — логика как в `hotel_scraper.py` (url showcaptcha/checkcaptcha, текст “Вы человек”, iframe captcha)
- `_wait_for_manual_captcha()` — печатает сообщение и ждёт Enter

капча проверяется:
- после первого захода
- после попытки редиректа на `/reviews/`
- после каждой прокрутки

### 5.5 поиск контейнера скролла и стратегия скролла
контейнер прокрутки:
- `.scroll__content` и варианты (`[class*='scroll__content']`)

метод `_find_scroll_container()` пытается найти container, если не нашёл, скроллит весь `page`.

метод `_scroll_to_load_all(container)`:
- считает текущее число карточек отзывов (`_count_reviews()`)
- если число не растёт → увеличивает `idle_rounds`
- остановка при `max_idle_rounds = 4`
- если задан `max_reviews` и достигнут лимит → остановка

скролл:
- если `container` найден:
  - `container.evaluate("el => el.scrollTo(0, el.scrollHeight)")`
- иначе:
  - `page.mouse.wheel(0, 50000)`

после каждого скролла:
- `wait_for_timeout(900)`
- `_try_click_common_buttons()`
- `_wait_for_manual_captcha()`

### 5.6 селекторы и извлечение полей отзыва
карточки отзывов:
- `.business-review-view` и `[class*='business-review-view']`

текст:
- `.spoiler-view__text-container`
- `.business-review-view__body-text`
- `.business-review-view__body`

автор:
- `.business-review-view__author`
- `.business-review-view__author-name`
- `a[href*='user']`

дата:
- приоритетно берётся из `<time>`:
  - атрибут `datetime` (если есть) или текст
- fallback: `.business-review-view__date`, `time`, `[class*='date']`

рейтинг:
- сначала `aria-label` с “Оценка”
- затем `meta[itemprop='ratingValue']` (attr `content`)
- затем текстовые селекторы:
  - `.business-review-view__rating`
  - `[aria-label*='Оценка']`
  - `meta[itemprop='ratingValue']`
  - `[itemprop='ratingValue']`

парсинг рейтинга:
- regex `([1-5](?:[\.,]\d)?)`, допускает “4,7” и “4.7”

### 5.7 дедупликация отзывов
После извлечения всех карточек:
- формируется ключ `(author, date, text)`
- если ключ уже встречался → отзыв пропускается

### 5.8 Sync wrapper
Как и в `hotel_scraper.py`, `main()` делает sync API поверх async с учётом уже запущенного event loop (через `ThreadPoolExecutor`)

---

## 6) `main.py` — оркестрация и схема сырых данных

`main.py` выполняет последовательно:
1) `hotel_scraper.main("output/hotels.json")`
2) читает список отелей из `output/hotels.json`
3) для каждого отеля вызывает `reviews_scraper.main(url, structured=True, headless=False, max_reviews=60, user_data_dir="output/pw_profile_reviews")`
4) пишет `output/reviews_raw.csv`

`output/reviews_raw.csv` колонки:
- `hotel`
- `district`
- `source` (фиксировано `yandex_maps`)
- `author`
- `date`
- `rating`
- `review`

---

## 7. NLP-пайплайн (`nlp_hotels.py`)

### 7.1 очистка текста
функция `clean_text()` нормализует исходный отзыв:
- удаление HTML-тегов
- удаление URL
- нормализация пробелов/переносов

Выход: `text_clean`

### 7.2 парсинг даты и сезонность
- `parse_date_ru()` распознаёт относительные даты (“вчера”, “N дней назад”) и форматы `dd.mm.yyyy`, `dd <месяц> [yyyy]`
- `season_from_date()` переводит дату в сезон: `winter/spring/summer/autumn`
- сохранение `date_iso` (если распарсилось)

### 7.3 лингвистическая разметка (spaCy)
обработка идёт батчами через `nlp.pipe()` (важно для производительности)

сохраняются:
- `tokens_json` — список токенов
- `lemmas` — строка лемм (через пробел)
- `pos_json` — POS-теги по токенам
- `morph_json` — морфологические признаки
- `deps_json` — зависимости (индекс токена, индекс головы, тип связи)
- `entities_json` — NER сущности (text/label) + доменная нормализация/фильтры (если включены в коде)

#### лемматизация RU: spaCy + fallback pymorphy2
если spaCy вернула лемму, совпадающую с токеном (частая ситуация в RU), используется `pymorphy2.normal_form` как fallback.

### 7.4 тональность (Transformers, RuBERT)
по умолчанию включена
- pipeline: `text-classification`
- max_length=256, truncation=True
выход:
- `sentiment` (label)
- `sentiment_score` (confidence)

отключение:
- `--no_sentiment`

### 7.5 (опционально) перевод RU→EN
включение:
- `--translate`
выход:
- `translation_en` (str)

---

## 8. темы и аспекты (rule-based)

### 8.1 Topic classification
функция классифицирует отзыв по словарям `TOPIC_KEYWORDS`:
- пересечение лемм отзыва с ключевыми словами темы
- выбирается тема с максимальным score

выход: `topic`

### 8.2 Aspects detection
функция возвращает список аспектов по словарям `ASPECTS`:
- аспект считается присутствующим, если пересечение лемм не пустое

выход: `aspects_json` (JSON-строка списка аспектов)

---

## 9. итоговые таблицы (`--output`, по умолчанию `out/`)

### 9.1 `reviews_typed.csv`


ключевые поля:
- `review_id` — стабильный id (sha1 от hotel/author/date/text_clean, укороченный)
- `hotel`, `district`, `author`, `date`, `date_iso`, `season`, `rating`, `source`
- `text_raw`, `text_clean`
- `tokens_json`, `lemmas`, `pos_json`, `morph_json`, `deps_json`, `entities_json`
- `sentiment`, `sentiment_score` (если не отключено)
- `translation_en` (если включено)
- `topic`, `aspects_json`

> CSV плоский, поэтому структурные поля сериализуются в JSON-строки

### 9.2 `word_stats.csv`
частоты лемм по срезам:
- `hotel`, `district`, `season`, `sentiment`, `lemma`, `count`

### 9.3 `topics.csv`
число отзывов по темам:
- `hotel`, `district`, `season`, `sentiment`, `topic`, `reviews_count`

### 9.4 `entities.csv`
частоты NER сущностей:
- `hotel`, `district`, `season`, `sentiment`, `entity_label`, `entity_text`, `count`

### 9.5 `aspect_sentiment.csv`
аспекты × тональность:
- `hotel`, `district`, `season`, `sentiment`, `aspect`, `count`

### 9.6 `complaints.csv` (униграммы + dependency-фразы)
таблица “жалоб” строится из **NEGATIVE** отзывов (требует sentiment, иначе NEGATIVE не определены).

колонки:
- `hotel`, `district`, `season`
- `n` — размер терма (1 = униграмма, 2 = фраза/“биграмма”)
- `term` — лемма или фраза
- `count` — частота

#### извлечение термов
- `n=1`: униграммы = леммы токенов (после фильтрации stopwords)
- `n=2`: фразы извлекаются **по синтаксическим зависимостям spaCy**, а не как “соседние слова”:
  - `amod(ADJ→NOUN/PROPN)`: `грязный номер`
  - `nmod/compound/appos(NOUN↔NOUN)`: `шум дороги`
  - `neg(не→head)`: `не работать`
  - (ограниченно) `advmod(ADV→head)` для списка усилителей/оценок: `слишком шумно`, `плохо работать`

фильтрация по порогу:
- `min_count` применяется после агрегации по ключам `(hotel, district, season, n, term)`.

> настройка `min_count` задаётся в `export_outputs()` (вызов `build_complaints(...)`).

### 9.7 `tfidf_hotel.csv`, `tfidf_district.csv`, `tfidf_season.csv`
TF‑IDF “характерных слов” по группам.

схема:
- `group_field` — поле группировки (`hotel`/`district`/`season`)
- `group` — значение группы
- `term`, `tfidf`, `tf`, `df`, `total_terms`

IDF используется со сглаживанием:
- `idf = log((N + 1)/(df + 1)) + 1`

---

## 10. производительность и ресурсы

- spaCy ускоряется через `nlp.pipe(..., batch_size=...)`
- тональность (Transformers) и перевод (NLLB) наиболее затратны
- На macOS возможен запуск Torch на `mps` (Apple Silicon), что ускоряет inference, но может менять предупреждения/совместимость

---

## 11. Диагностика и типовые проблемы

### 11.1 Playwright/Chromium
- проверьте установку браузера: `playwright install chromium`
- если страницы не грузятся/селекторы не находятся: верстка Yandex Maps могла измениться

### 11.2 spaCy model not found
```bash
python -m spacy download ru_core_news_sm
```

### 11.3 NLTK stopwords отсутствуют
```bash
python -c "import nltk; nltk.download('stopwords')"
```

### 11.4 LibreSSL предупреждение (urllib3 v2)
на macOS с системным Python иногда появляется предупреждение про LibreSSL vs OpenSSL. 
рекомендуется использовать Python из `pyenv`/`conda` или собрать окружение с OpenSSL 1.1.1+

---

## 12. Конфигурация / точки расширения

- расширение тем: `TOPIC_KEYWORDS`
- расширение аспектов: `ASPECTS`
- нормализация сущностей (при необходимости): словари/фильтры в блоке NER post-processing
- улучшение “жалоб”: добавить шаблоны по deps (например `obj`, `obl`) или ABSA-lite (аспект + оценка через dependency links)

---

## 13. примечания по данным и этике
- источник данных: публично доступные отзывы на Yandex Maps
- использование данных должно соответствовать правилам источника и требованиям проекта (в т.ч. по персональным данным/PII)


