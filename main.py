from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import hotel_scraper
import reviews_scraper


JSON_HOTELS = "output/hotels.json"
CSV_REVIEWS = "output/reviews_raw.csv"


def load_hotels(json_file_name: str) -> list[dict[str, Any]]:
    with open(json_file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("hotels.json должен быть списком объектов")
    return data


def write_reviews_csv(rows: list[dict[str, Any]], csv_path: str) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "hotel",
        "district",
        "source",
        "author",
        "date",
        "rating",
        "review",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    Path("output").mkdir(parents=True, exist_ok=True)

    hotel_scraper.main(JSON_HOTELS)

    hotels = load_hotels(JSON_HOTELS)

    rows: list[dict[str, Any]] = []

    for i, h in enumerate(hotels, start=1):
        hotel_name = str(h.get("title", "")).strip()
        url = str(h.get("link", "")).strip()
        district = str(h.get("district", "")).strip()

        if not hotel_name or not url:
            continue

        print(f"\n[{i}/{len(hotels)}] {hotel_name} | {district}")
        print(url)

        try:
            reviews = reviews_scraper.main(
                url,
                logging=False,
                structured=True,
                headless=False,
                max_reviews=60,
                user_data_dir="output/pw_profile_reviews",
            )
        except Exception as e:
            print(f"Ошибка при сборе отзывов: {e}")
            continue

        for r in reviews:
            text = str(r.get("review", "") or "").strip()
            if not text:
                continue

            rows.append(
                {
                    "hotel": hotel_name,
                    "district": district,
                    "source": "yandex_maps",
                    "author": r.get("author", ""),
                    "date": r.get("date", ""),
                    "rating": r.get("rating", ""),
                    "review": text,
                }
            )

    write_reviews_csv(rows, CSV_REVIEWS)
    print(f"\nOK: сохранено {len(rows)} отзывов -> {CSV_REVIEWS}")


if __name__ == "__main__":
    main()
