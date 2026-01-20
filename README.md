# анализ отзывов об отелях Нижний Новгород

проект собирает отзывы об отелях из Yandex Maps, очищает текст, удаляет стоп-слова, лемматизирует, выделяет темы и готовит таблицы для визуализации (например, в Yandex DataLens)

## цели
- собрать отзывы для заданного типа объектов (отели)
- подготовить структуры данных и единый формат хранения
- исключить стоп-слова, сгруппировать леммы (лемматизация)
- посчитать частоты слов
- построить визуализации (графики/таблицы) и собрать дашборд в DataLens
- ответить на вопросы:
  - какие слова встречаются чаще всего (в целом / по сезонам)?
  - есть ли различия в частоте слов между позитивными и негативными отзывами?
  - какие темы чаще всего упоминают пользователи?

---

## архитектура пайплайна

1. **парсинг списка отелей** (`hotel_scraper.py`)
   - ищет отели по районам Нижнего Новгорода в Yandex Maps
   - сохраняет `output/hotels.json` (title/link/district)

2. **парсинг отзывов** (`reviews_scraper.py`)
   - открывает страницу отзывов каждого отеля
   - прокручивает список отзывов, собирает:
     - `review` (текст),
     - `date` (дата на странице),
     - `rating` (оценка),
     - `author` (если доступно)
   - работает через Playwright **Async API**.
   - поддерживает “ручное” прохождение капчи (скрипт ждёт, пока пользователь решит её в браузере)

3. **склейка и сохранение сырых данных** (`main.py`)
   - запускает сбор отелей и сбор отзывов
   - сохраняет общий CSV: `output/reviews_raw.csv`

4. **NLP обработка + агрегаты под визуализации** (`hotel_nlp.py`)
   - очистка текста, стоп-слова
   - лемматизация (русский, плюс поддержка латиницы для слов вроде `wifi`)
   - расчёт:
     - `season` (по дате),
     - `sentiment` (по рейтингу: positive/negative/neutral),
     - `topic` (простая тематическая классификация по словарям)
   - выгрузки в `out/`:
     - `reviews_typed.csv` (строка = отзыв),
     - `counts_typed.csv` (частоты по отелям),
     - `word_stats.csv` (частоты лемм по season/sentiment),
     - `topics.csv` (темы по season/sentiment)

---

## структура проекта
├── hotel_scraper.py \
├── reviews_scraper.py \
├── main.py \
├── hotel_nlp.py \
├── output/ \
├ ├── hotels.json \
├ ├── reviews_raw.csv \
├── out/ \
├ ├── reviews_typed.csv \
├ ├── counts_typed.csv \
├ ├── word_stats.csv \
├ ├── topics.csv


---

## установка и зависимости

### 1) виртуальное окружение (рекомендуется)

```bash

  python -m venv venv
  # Windows: venv\Scripts\activate
  source venv/bin/activate
```

### 2) установка библиотек

```bash
  pip install pandas numpy pymorphy2 pymorphy2-dicts-ru playwright
```

### 3) установка браузера для Playwright

```bash
  playwright install chromium
  #если используете NLTK-стопслова (опционально)
  pip install nltk
  python -c "import nltk; nltk.download('stopwords')"
```


---

## запуск 

### сбор отелей и отзывов 

```bash
  python main.py
```
#### результат 

- `outpt/hotels.json`
-  `output/reviews_raw.csv`

> важно: при появлении капчи "вы человек" браузер откроется в видимом режиме  решите капчу вручную и продолжайте сбор

### NLP обработка и подготовка таблиц

```bash
  python hotel_nlp.py --input output/reviews_raw.csv --output out
```

#### результат 
- `out/reviews_typed.csv`
- `out/word_stats.csv`
- `out/topiccs.csv`
- `counts_typed.csv`

---


## форматы данных

### `output/reviews_raw.csv`

#### колонки: 

- `hotel` - название отеля
- `district` - район
- `source` - источник/URL
- `author` - автор 
- `date` - дата отзыва
- `rating` - оценка
- `review` - текст отзыва

### `out/reviews_typed.csv`

#### колонки: 

- `review_id` - id отзыва
- `hotel` - название отеля
- `district` - район
- `source` - источник/URL
- `author` - автор 
- `date` - дата отзыва
- `rating` - оценка
- `text_raw` - текст отзыва
- `season` - сезон
- `sentiment` - positive/negative/neutral

### `out/words_stats.csv`

#### колонки: 

- `season` - сезон
- `sentiment` - positive/negative/neutral
- `lemma` - слово
- `count` - количество слов

### `out/topics.csv`

#### колонки: 

- `season` - сезон
- `sentiment` - positive/negative/neutral
- `topic` - раздел
- `count` - количество упоминаний данного раздела

### `out/сcount_typed.csv`

#### колонки: 

- `hotel` - отель
- `lemma` - слово
- `count` - количество слов


---


## ограничения 

- Yandex Maps может показывать капчу; проект рассчитан на ручное подтверждение
- структура страниц может меняться, что потребует обновления селекторов
- дата может быть относительной ("вчера", "2 дня назад") и не всегда парсится в сезон
- сентимент считается по рейтингу (простая эвристика), а не по NLP-модели
- темы выделяются по словарям (объяснимо и стабильно), а не topic modeling

