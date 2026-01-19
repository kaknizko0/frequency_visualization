from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from playwright.async_api import (
    Error as PlaywrightError,
    Page,
    ElementHandle,
    async_playwright,
)


_RE_WS = re.compile(r"\s+")
_RE_RATING = re.compile(r"([1-5](?:[\.,]\d)?)")


def _norm(text: Optional[str]) -> str:
    if not text:
        return ""
    return _RE_WS.sub(" ", text).strip()


def to_reviews_url(url: str) -> str:

    u = (url or "").strip()
    if not u:
        return u

    u = u.split("?")[0]

    if "/gallery" in u:
        u = re.sub(r"/gallery/?$", "/reviews/", u)

    if re.search(r"/reviews/?$", u):
        if not u.endswith("/"):
            u += "/"
        return u

    if not u.endswith("/"):
        u += "/"
    return u + "reviews/"


def _parse_rating(raw: str) -> Optional[float]:
    raw = _norm(raw)
    if not raw:
        return None
    m = _RE_RATING.search(raw)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except ValueError:
        return None


@dataclass
class Review:
    text: str
    date: str = ""
    rating: Optional[float] = None
    author: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "review": self.text,
            "date": self.date,
            "rating": self.rating,
            "author": self.author,
        }


class AsyncReviewsScraper:

    def __init__(
        self,
        url: str,
        logging: bool = False,
        headless: bool = False,
        slow_mo: int = 100,
        timeout_ms: int = 60_000,
        max_reviews: Optional[int] = None,
        user_data_dir: str = "output/pw_profile_reviews",
    ):
        self._url = to_reviews_url(url)
        self._logging = logging
        self._headless = headless
        self._slow_mo = slow_mo
        self._timeout_ms = timeout_ms
        self._max_reviews = max_reviews
        self._user_data_dir = user_data_dir

        self._playwright = None
        self._context = None
        self._page: Optional[Page] = None

        self._scroll_container_selectors = [
            ".scroll__content",
            "[class*='scroll__content']",
        ]
        self._review_card_selectors = [
            ".business-review-view",
            "[class*='business-review-view']",
        ]

        self._text_selectors = [
            ".spoiler-view__text-container",
            ".business-review-view__body-text",
            ".business-review-view__body",
        ]
        self._author_selectors = [
            ".business-review-view__author",
            ".business-review-view__author-name",
            "a[href*='user']",
        ]
        self._date_selectors = [
            ".business-review-view__date",
            "time",
            "[class*='date']",
        ]
        self._rating_selectors = [
            ".business-review-view__rating",
            "[aria-label*='Оценка']",
            "meta[itemprop='ratingValue']",
            "[itemprop='ratingValue']",
        ]

    def _log(self, msg: str) -> None:
        if self._logging:
            print(msg)

    async def init_playwright(self) -> None:
        Path(self._user_data_dir).parent.mkdir(parents=True, exist_ok=True)
        self._playwright = await async_playwright().start()

        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=self._user_data_dir,
            headless=self._headless,
            slow_mo=self._slow_mo,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
            locale="ru-RU",
        )

        self._page = await self._context.new_page()
        self._page.set_default_timeout(self._timeout_ms)

    async def close(self) -> None:
        try:
            if self._context:
                await self._context.close()
        except Exception:
            pass
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass

    async def _try_click_common_buttons(self) -> None:
        assert self._page is not None
        candidates = [
            "text=Принять",
            "text=Согласен",
            "text=ОК",
            "text=Понятно",
            "text=Закрыть",
        ]
        for sel in candidates:
            try:
                loc = self._page.locator(sel)
                if await loc.count() > 0:
                    await loc.first.click(timeout=1500)
                    self._log(f"Нажали кнопку: {sel}")
            except Exception:
                continue

    async def _looks_like_captcha(self) -> bool:
        assert self._page is not None
        url = self._page.url or ""
        if "showcaptcha" in url or "checkcaptcha" in url:
            return True

        try:
            if await self._page.locator("text=Вы человек").count() > 0:
                return True
        except Exception:
            pass
        try:
            if await self._page.locator("iframe[src*='captcha']").count() > 0:
                return True
        except Exception:
            pass
        return False

    async def _wait_for_manual_captcha(self) -> None:
        assert self._page is not None
        if not await self._looks_like_captcha():
            return

        print("\n[CAPTCHA] Похоже, Яндекс показал проверку 'Вы человек'.")
        print("Реши её в открывшемся окне браузера, затем вернись в терминал.")

        try:
            await asyncio.to_thread(input, "Нажми Enter после того, как решишь капчу... ")
        except Exception:
            await self._page.wait_for_timeout(30_000)

    async def _goto(self) -> None:
        assert self._page is not None

        await self._page.goto(self._url, wait_until="domcontentloaded")
        await self._page.wait_for_timeout(500)

        await self._try_click_common_buttons()
        await self._wait_for_manual_captcha()

        if not re.search(r"/reviews/?$", (self._page.url or "")):
            await self._page.goto(to_reviews_url(self._page.url), wait_until="domcontentloaded")
            await self._page.wait_for_timeout(500)
            await self._try_click_common_buttons()
            await self._wait_for_manual_captcha()

    async def _find_scroll_container(self) -> Optional[ElementHandle]:
        assert self._page is not None
        for sel in self._scroll_container_selectors:
            try:
                el = await self._page.query_selector(sel)
                if el:
                    return el
            except Exception:
                continue
        return None

    async def _count_reviews(self) -> int:
        assert self._page is not None
        for sel in self._review_card_selectors:
            try:
                cards = await self._page.query_selector_all(sel)
                if cards:
                    return len(cards)
            except Exception:
                continue
        return 0

    async def _scroll_to_load_all(self, container: Optional[ElementHandle]) -> None:
        assert self._page is not None

        prev = -1
        idle_rounds = 0
        max_idle_rounds = 4

        while idle_rounds < max_idle_rounds:
            current = await self._count_reviews()

            if self._max_reviews is not None and current >= self._max_reviews:
                self._log(f"Достигнут лимит max_reviews={self._max_reviews}")
                break

            if current == prev:
                idle_rounds += 1
            else:
                idle_rounds = 0
                prev = current

            self._log(f"Отзывы: {current}. Прокрутка...")

            try:
                if container:
                    await container.evaluate("el => el.scrollTo(0, el.scrollHeight)")
                else:
                    await self._page.mouse.wheel(0, 50_000)
            except Exception:
                try:
                    await self._page.mouse.wheel(0, 50_000)
                except Exception:
                    pass

            await self._page.wait_for_timeout(900)
            await self._try_click_common_buttons()
            await self._wait_for_manual_captcha()

    async def _first_text(self, root: ElementHandle, selectors: list[str]) -> str:
        for sel in selectors:
            try:
                el = await root.query_selector(sel)
                if el:
                    txt = _norm(await el.text_content())
                    if txt:
                        return txt
            except Exception:
                continue
        return ""

    async def _first_attr(self, root: ElementHandle, selectors: list[str], attr: str) -> str:
        for sel in selectors:
            try:
                el = await root.query_selector(sel)
                if el:
                    val = _norm(await el.get_attribute(attr))
                    if val:
                        return val
            except Exception:
                continue
        return ""

    async def _get_review_cards(self) -> list[ElementHandle]:
        assert self._page is not None
        for sel in self._review_card_selectors:
            try:
                cards = await self._page.query_selector_all(sel)
                if cards:
                    return cards
            except Exception:
                continue
        return []

    async def _extract_one_review(self, card: ElementHandle) -> Review:
        text = await self._first_text(card, self._text_selectors)
        author = await self._first_text(card, self._author_selectors)

        date = ""
        try:
            time_el = await card.query_selector("time")
            if time_el:
                date = _norm(await time_el.get_attribute("datetime")) or _norm(await time_el.text_content())
        except Exception:
            date = ""
        if not date:
            date = await self._first_text(card, self._date_selectors)

        raw_rating = (
            await self._first_attr(card, ["[aria-label*='Оценка']"], "aria-label")
            or await self._first_attr(card, ["meta[itemprop='ratingValue']"], "content")
            or await self._first_text(card, self._rating_selectors)
        )
        rating = _parse_rating(raw_rating)

        return Review(text=text, date=date, rating=rating, author=author)

    async def scrape(self) -> list[Review]:
        if not self._url:
            raise ValueError("URL не задан")
        if self._page is None:
            raise RuntimeError("Playwright не инициализирован. Вызови init_playwright().")

        await self._goto()

        container = await self._find_scroll_container()
        if not container:
            self._log("Контейнер прокрутки не найден, скроллим страницу целиком.")

        await self._scroll_to_load_all(container)

        cards = await self._get_review_cards()
        self._log(f"Карточек отзывов найдено: {len(cards)}")

        reviews: list[Review] = []
        seen: set[tuple[str, str, str]] = set()

        for card in cards:
            r = await self._extract_one_review(card)
            if not r.text:
                continue
            key = (r.author, r.date, r.text)
            if key in seen:
                continue
            seen.add(key)
            reviews.append(r)

            if self._max_reviews is not None and len(reviews) >= self._max_reviews:
                break

        return reviews


async def main_async(
    url: str,
    logging: bool = False,
    structured: bool = False,
    headless: bool = False,
    slow_mo: int = 100,
    timeout_ms: int = 60_000,
    max_reviews: Optional[int] = None,
    user_data_dir: str = "output/pw_profile_reviews",
):
    scraper = AsyncReviewsScraper(
        url=url,
        logging=logging,
        headless=headless,
        slow_mo=slow_mo,
        timeout_ms=timeout_ms,
        max_reviews=max_reviews,
        user_data_dir=user_data_dir,
    )

    await scraper.init_playwright()
    try:
        reviews = await scraper.scrape()
    finally:
        await scraper.close()

    if logging:
        for i, r in enumerate(reviews, start=1):
            print(f"[{i}] rating={r.rating} date={r.date} author={r.author} | {r.text}")

    if structured:
        return [r.as_dict() for r in reviews]
    return [r.text for r in reviews]


def main(
    url: str,
    logging: bool = False,
    structured: bool = False,
    headless: bool = False,
    slow_mo: int = 100,
    timeout_ms: int = 60_000,
    max_reviews: Optional[int] = None,
    user_data_dir: str = "output/pw_profile_reviews",
):

    try:
        asyncio.get_running_loop()
        loop_running = True
    except RuntimeError:
        loop_running = False

    if not loop_running:
        return asyncio.run(
            main_async(
                url=url,
                logging=logging,
                structured=structured,
                headless=headless,
                slow_mo=slow_mo,
                timeout_ms=timeout_ms,
                max_reviews=max_reviews,
                user_data_dir=user_data_dir,
            )
        )

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(
            lambda: asyncio.run(
                main_async(
                    url=url,
                    logging=logging,
                    structured=structured,
                    headless=headless,
                    slow_mo=slow_mo,
                    timeout_ms=timeout_ms,
                    max_reviews=max_reviews,
                    user_data_dir=user_data_dir,
                )
            )
        )
        return fut.result()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python reviews_scraper.py <url>")
        raise SystemExit(2)

    data = main(sys.argv[1], logging=True, structured=True, headless=False)
    print(f"\nTotal reviews: {len(data)}")
