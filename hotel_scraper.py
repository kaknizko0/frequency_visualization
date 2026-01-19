from __future__ import annotations

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Any
from urllib.parse import quote

from playwright.async_api import async_playwright, Page, ElementHandle

CITY_CODE = 47
CITY_SLUG = "nizhny-novgorod"

DISTRICTS = [
    "Автозаводский",
    "Канавинский",
    "Ленинский",
    "Московский",
    "Нижегородский",
    "Приокский",
    "Советский",
    "Сормовский",
]

BASE_URL = "https://yandex.ru"

BROWSER_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
]


def get_url(district: Optional[str] = None) -> str:
    base = f"{BASE_URL}/maps/{CITY_CODE}/{CITY_SLUG}/search/"

    if district:
        query = f"гостиницы {district} район"
    else:
        query = "гостиницы"

    return base + quote(query)


def normalize_org_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u

    if u.startswith("/"):
        u = BASE_URL + u

    u = u.split("?")[0]

    u = re.sub(r"/(gallery|reviews|photos|services|menu|prices|news|posts)/?$", "/", u)

    if not u.endswith("/"):
        u += "/"
    return u


async def _looks_like_captcha(page: Page) -> bool:
    url = page.url or ""
    if "showcaptcha" in url or "checkcaptcha" in url:
        return True
    try:
        if await page.locator("text=Вы человек").count() > 0:
            return True
    except Exception:
        pass
    try:
        if await page.locator("iframe[src*='captcha']").count() > 0:
            return True
    except Exception:
        pass
    return False


async def wait_for_manual_captcha(page: Page) -> bool:
    if not await _looks_like_captcha(page):
        return True

    print("\n[CAPTCHA] Похоже, Яндекс показал проверку 'Вы человек'.")
    print("Реши её в открывшемся окне браузера, затем вернись в терминал.")

    try:
        await asyncio.to_thread(input, "Нажми Enter после решения капчи... ")
    except Exception:
        await page.wait_for_timeout(30_000)

    return not await _looks_like_captcha(page)


async def _scroll_results(page: Page) -> None:
    last_count = 0
    stagnation = 0
    max_stagnation = 4

    while stagnation < max_stagnation:
        try:
            items = await page.query_selector_all("li[class*='search-snippet'], .search-business-snippet-view, li.search-business-snippet-view")
        except Exception:
            items = []

        current = len(items)
        if current > last_count:
            last_count = current
            stagnation = 0
        else:
            stagnation += 1

        try:
            container = await page.query_selector(".scroll__content")
            if container:
                await container.hover()
            else:
                await page.hover("body")
        except Exception:
            pass

        try:
            await page.mouse.wheel(0, 12_000)
        except Exception:
            pass

        await page.wait_for_timeout(700)


async def _extract_hotels_from_page(page: Page, district: str) -> list[dict[str, Any]]:
    items = await page.query_selector_all(".search-business-snippet-view")

    hotels: list[dict[str, Any]] = []

    title_selectors = [
        "[class*='search-business-snippet-view__title']",
        "[class*='org-title']",
        "[class*='title']",
    ]
    link_selectors = [
        "a[href*='/org/']",
        "a[class*='link']",
    ]

    for item in items:
        title = ""
        for sel in title_selectors:
            try:
                el = await item.query_selector(sel)
                if el:
                    title = (await el.text_content() or "").strip()
                    if title:
                        break
            except Exception:
                continue
        if not title:
            continue

        href = ""
        for sel in link_selectors:
            try:
                el = await item.query_selector(sel)
                if el:
                    href = (await el.get_attribute("href") or "").strip()
                    if href:
                        break
            except Exception:
                continue

        if not href:
            continue

        url = normalize_org_url(href)

        hotels.append(
            {
                "title": title,
                "link": url,
                "district": district,
                "city": "Нижний Новгород",
            }
        )

    return hotels


async def scrape_async(
    max_districts: Optional[int] = None,
    headless: bool = False,
    slow_mo: int = 100,
    timeout_ms: int = 60_000,
    user_data_dir: str = "output/pw_profile_maps",
) -> list[dict[str, Any]]:
    Path(user_data_dir).parent.mkdir(parents=True, exist_ok=True)

    hotels_all: list[dict[str, Any]] = []
    seen: set[str] = set()

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=headless,
            slow_mo=slow_mo,
            args=BROWSER_ARGS,
            locale="ru-RU",
        )
        page = await context.new_page()
        page.set_default_timeout(timeout_ms)

        districts = DISTRICTS if max_districts is None else DISTRICTS[:max_districts]

        for district in districts:
            print(f"\nСбор данных для района: {district}")
            url = get_url(district)
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_timeout(500)

            if not await wait_for_manual_captcha(page):
                print(f"Пропускаем район '{district}' (капча не решена).")
                continue

            await _scroll_results(page)

            hotels = await _extract_hotels_from_page(page, district)
            print(f"🏨 Найдено отелей: {len(hotels)}")

            for h in hotels:
                link = h.get("link", "")
                if link and link not in seen:
                    seen.add(link)
                    hotels_all.append(h)

            await page.wait_for_timeout(800)

        await context.close()

    return hotels_all


def scrape(
    max_districts: Optional[int] = None,
    headless: bool = False,
    slow_mo: int = 100,
    timeout_ms: int = 60_000,
    user_data_dir: str = "output/pw_profile_maps",
) -> list[dict[str, Any]]:

    async def _runner():
        return await scrape_async(
            max_districts=max_districts,
            headless=headless,
            slow_mo=slow_mo,
            timeout_ms=timeout_ms,
            user_data_dir=user_data_dir,
        )

    try:
        asyncio.get_running_loop()
        loop_running = True
    except RuntimeError:
        loop_running = False

    if not loop_running:
        return asyncio.run(_runner())

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(lambda: asyncio.run(_runner()))
        return fut.result()


def main(
    json_file_name: str = "output/hotels.json",
    max_districts: Optional[int] = None,
) -> None:
    Path(json_file_name).parent.mkdir(parents=True, exist_ok=True)

    hotels = scrape(max_districts=max_districts, headless=False)

    with open(json_file_name, "w", encoding="utf-8") as f:
        json.dump(hotels, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено {len(hotels)} отелей в файл '{json_file_name}'")


if __name__ == "__main__":
    main()
