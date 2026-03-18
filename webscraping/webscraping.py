import base64
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlparse
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerationConfig
from vertexai.generative_models import GenerativeModel


BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "")
IS_CLOUD_RUNTIME = bool(os.environ.get("K_SERVICE") or os.environ.get("CLOUD_RUN_JOB") or os.environ.get("FUNCTION_TARGET"))
OUTPUT_DIR = "/tmp/data" if IS_CLOUD_RUNTIME else os.path.join(os.path.dirname(__file__), "data")
DATASTORE_JSON = os.path.join(OUTPUT_DIR, "activities_latest.json")
DATASTORE_CSV = os.path.join(OUTPUT_DIR, "activities_latest.csv")
DATASTORE_JSONL = os.path.join(OUTPUT_DIR, "activities_latest_rag.jsonl")
RUN_REPORT_JSON = os.path.join(OUTPUT_DIR, "ingestion_report.json")

ENABLE_LLM_TAGGING = os.environ.get("ENABLE_LLM_TAGGING", "false").lower() in {"1", "true", "yes"}
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT", "")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
VERTEX_MODEL = os.environ.get("VERTEX_MODEL", "gemini-3.1-pro-preview")
LLM_TAGGING_MAX_CALLS = int(os.environ.get("LLM_TAGGING_MAX_CALLS", "120"))

EXPECTED_COLUMNS = [
    "activity_id",
    "title",
    "start_datetime_raw",
    "start_datetime_iso",
    "location",
    "description",
    "url",
    "source",
    "source_confidence",
    "status",
    "last_seen_at",
]

SOURCES = [
    {
        "name": "eventbrite",
        "url": "https://www.eventbrite.sg/d/singapore--singapore/seniors/",
        "parser": "parse_eventbrite",
    },
    {
        "name": "agewellsg",
        "url": "https://www.agewellsg.gov.sg/active-ageing/",
        "parser": "parse_agewellsg",
    },
    {
        "name": "lionsbefrienders",
        "url": "https://www.lionsbefrienders.org.sg/",
        "parser": "parse_lionsbefrienders",
    },
    {
        "name": "onepa",
        "url": "https://www.onepa.gov.sg/events",
        "parser": "parse_onepa",
    },
    {
        "name": "meetup",
        "url": "https://www.meetup.com/find/sg--singapore/seniors/",
        "parser": "parse_meetup",
    },
    {
        "name": "timeoutsg",
        "url": "https://www.timeout.com/singapore/things-to-do",
        "parser": "parse_timeoutsg",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

CATEGORY_OPTIONS = ["art", "dance", "wellness", "social", "learning", "general"]
CATEGORY_PATTERNS = {
    "art": ["art", "arts", "museum", "gallery", "exhibition", "painting", "pottery", "craft"],
    "dance": ["dance", "salsa", "jive", "cha cha", "zumba", "ballroom"],
    "wellness": ["wellness", "qigong", "breathwork", "yoga", "therapy", "health", "mindfulness"],
    "social": ["social", "meet", "community", "seniors-meet-seniors", "friendship", "networking"],
    "learning": ["workshop", "talk", "class", "course", "lecture", "insights"],
}

_vertex_model = None
_llm_category_cache = {}
_llm_calls_made = 0
_llm_tag_disabled_reason = ""
_llm_tag_parse_failures = 0
LLM_PARSE_FAILURE_DISABLE_THRESHOLD = int(os.environ.get("LLM_PARSE_FAILURE_DISABLE_THRESHOLD", "40"))


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def upload_to_gcs(local_file_path, destination_blob_name):
    if not BUCKET_NAME:
        return
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        print(f"Uploaded {local_file_path} to gs://{BUCKET_NAME}/{destination_blob_name}")
    except Exception as exc:
        print(f"[WARN] Failed to upload {local_file_path} to GCS: {exc}")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def normalize_space(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def get_vertex_model():
    global _vertex_model
    if _vertex_model is None:
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        _vertex_model = GenerativeModel(VERTEX_MODEL)
    return _vertex_model


def classify_categories_keyword(title, description, source, location):
    searchable_blob = " ".join([title, description, source, location]).lower()
    categories = []
    for category_name, category_keywords in CATEGORY_PATTERNS.items():
        if any(re.search(rf"\b{re.escape(keyword)}\b", searchable_blob) for keyword in category_keywords):
            categories.append(category_name)

    if not categories:
        return ["general"]
    return categories[:2]


def classify_categories_llm(title, description, source, location):
    global _llm_calls_made
    global _llm_tag_disabled_reason
    global _llm_tag_parse_failures

    if not ENABLE_LLM_TAGGING:
        return []
    if not VERTEX_PROJECT_ID:
        return []
    if _llm_calls_made >= LLM_TAGGING_MAX_CALLS:
        return []
    if _llm_tag_disabled_reason:
        return []

    cache_key = hashlib.sha1(
        "|".join([title, description, source, location]).encode("utf-8")
    ).hexdigest()
    if cache_key in _llm_category_cache:
        return _llm_category_cache[cache_key]

    prompt = (
        "You classify senior-friendly events in Singapore. "
        f"Pick 1 or 2 categories only from: {', '.join(CATEGORY_OPTIONS)}. "
        "Return only valid JSON, with no markdown and no extra text. "
        "Do not invent categories. "
        "Use semantic meaning, not raw keyword overlap. "
        "For example, 'martial arts' is not visual art by default unless context is about art/crafts/museums/exhibitions. "
        "If unsure, choose general. "
        "Return strict JSON only in this format: "
        '{"categories":["<category1>","<optional_category2>"]}. '
        f"Title: {title}\n"
        f"Description: {description}\n"
        f"Source: {source}\n"
        f"Location: {location}"
    )

    try:
        model = get_vertex_model()
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0,
                max_output_tokens=80,
                response_mime_type="application/json",
            ),
        )

        text = normalize_space(getattr(response, "text", ""))
        if not text and getattr(response, "candidates", None):
            candidates = response.candidates or []
            if candidates and getattr(candidates[0], "content", None) and candidates[0].content.parts:
                first_part = candidates[0].content.parts[0]
                text = normalize_space(getattr(first_part, "text", ""))
        if not text:
            return []

        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
            if fenced_match:
                try:
                    parsed = json.loads(fenced_match.group(1))
                except json.JSONDecodeError:
                    parsed = None

            if parsed is None:
                object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
                if object_match:
                    try:
                        parsed = json.loads(object_match.group(0))
                    except json.JSONDecodeError:
                        parsed = None

        if parsed is None:
            inferred = []
            lowered = text.lower()
            for option in CATEGORY_OPTIONS:
                if option != "general" and re.search(rf"\b{re.escape(option)}\b", lowered):
                    inferred.append(option)
            if not inferred:
                inferred = ["general"]

            deduped = []
            for category in inferred:
                if category not in deduped:
                    deduped.append(category)

            _llm_tag_parse_failures += 1
            if _llm_tag_parse_failures >= LLM_PARSE_FAILURE_DISABLE_THRESHOLD:
                _llm_tag_disabled_reason = "llm-non-json-output"

            result = deduped[:2]
            _llm_category_cache[cache_key] = result
            _llm_calls_made += 1
            return result

        categories = parsed.get("categories", []) if isinstance(parsed, dict) else []
        filtered = [c for c in categories if c in CATEGORY_OPTIONS and c != "general"]
        if not filtered and "general" in categories:
            filtered = ["general"]
        if not filtered:
            filtered = ["general"]

        deduped = []
        for category in filtered:
            if category not in deduped:
                deduped.append(category)

        result = deduped[:2]
        _llm_tag_parse_failures = 0
        _llm_category_cache[cache_key] = result
        _llm_calls_made += 1
        return result
    except Exception as exc:
        message = str(exc)
        if ("404" in message and "models" in message) or "NOT_FOUND" in message:
            _llm_tag_disabled_reason = "model-unavailable-or-no-access"
        print(f"[WARN] LLM tagging failed for '{title[:60]}': {exc}")
        return []


def clean_text_for_seniors_fallback(title, description):
    merged = normalize_space(f"{title}. {description}")
    if not merged:
        return "Senior-friendly activity in Singapore."

    cleaned = re.sub(r"\bWaitlist\b", "", merged, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d+\s*seats?\s*left\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\$\s*\d+(?:\.\d{1,2})?", "", cleaned)
    cleaned = re.sub(r"\bby\s+[A-Za-z0-9 .&'-]+(?=\s|$)", "", cleaned, flags=re.IGNORECASE)
    cleaned = normalize_space(cleaned)

    if len(cleaned) > 260:
        cleaned = cleaned[:257].rstrip(" ,.;:-") + "..."
    return cleaned


def build_deterministic_summary(title, date_text, location, description):
    title_text = normalize_space(title)
    date_value = normalize_space(date_text)
    location_value = normalize_space(location)
    description_text = clean_text_for_seniors_fallback("", description)

    parts = []
    if title_text:
        parts.append(title_text)
    if date_value:
        parts.append(f"Date: {date_value}")
    if location_value:
        parts.append(f"Location: {location_value}")
    if description_text:
        parts.append(description_text)

    summary = ". ".join([part for part in parts if part]).strip(" .")
    if not summary:
        summary = "Senior-friendly activity in Singapore."
    if len(summary) > 320:
        summary = summary[:317].rstrip(" ,.;:-") + "..."
    return summary


def safe_get(url, timeout=10, retries=2, backoff_sec=2.0):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            time.sleep(random.uniform(1.0, 3.0))
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_sec ** attempt)
    print(f"[WARN] Failed to fetch {url}: {last_error}")
    return None


def text_from_first(parent, selectors, default=""):
    for selector in selectors:
        element = parent.select_one(selector)
        if element:
            text = normalize_space(element.get_text(" ", strip=True))
            if text:
                return text
    return default


def extract_date_text(text):
    text = normalize_space(text)
    patterns = [
        r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s*\d{1,2}\s+\w+,\s*\d{1,2}:\d{2}\b",
        r"\b\d{1,2}\s+\w+\s+\d{4},\s*\d{1,2}:\d{2}\b",
        r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+\d{1,2}\s+\w+[, ]*\d{0,4}\b",
        r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+at\s+\d{1,2}:\d{2}\b",
        r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)(?:day)?\s+at\s+\d{1,2}:\d{2}\b",
        r"\b\d{1,2}\s+\w+\s+\d{4}\b",
        r"\b\d{2}-\d{2}-\d{4}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_space(match.group(0))
    return ""


def parse_to_iso_datetime(value):
    value = normalize_space(value)
    if not value:
        return ""

    meetup_match = re.search(
        r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s*(\w{3})\s+(\d{1,2})\s*[·\-]\s*(\d{1,2}:\d{2})\s*([AP]M)\s*SST$",
        value,
        flags=re.IGNORECASE,
    )
    if meetup_match:
        current_year = datetime.now(timezone.utc).year
        normalized = f"{meetup_match.group(1)}, {meetup_match.group(2)} {meetup_match.group(3)} {current_year} {meetup_match.group(4)} {meetup_match.group(5)}"
        parsed = pd.to_datetime(normalized, errors="coerce", utc=True)
        if not pd.isna(parsed):
            return parsed.isoformat()

    candidates = [value]
    current_year = datetime.now(timezone.utc).year

    if re.search(r"\b\d{1,2}\s+\w+,\s*\d{1,2}:\d{2}$", value):
        candidates.append(f"{value} {current_year}")

    if re.search(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s*\d{1,2}\s+\w+,\s*\d{1,2}:\d{2}$", value):
        candidates.append(f"{value} {current_year}")

    for candidate in candidates:
        parsed = pd.to_datetime(candidate, errors="coerce", dayfirst=True, utc=True)
        if not pd.isna(parsed):
            return parsed.isoformat()

    weekday_match = re.search(
        r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+at\s+(\d{1,2}):(\d{2})$",
        value,
        flags=re.IGNORECASE,
    )
    if not weekday_match:
        return ""

    weekday_name = weekday_match.group(1).lower()[:3]
    hour = int(weekday_match.group(2))
    minute = int(weekday_match.group(3))
    weekday_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    target_day = weekday_map[weekday_name]

    now = datetime.now(timezone.utc)
    day_delta = (target_day - now.weekday()) % 7
    if day_delta == 0 and (hour, minute) <= (now.hour, now.minute):
        day_delta = 7

    next_date = now + pd.Timedelta(days=day_delta)
    resolved = datetime(
        next_date.year,
        next_date.month,
        next_date.day,
        hour,
        minute,
        tzinfo=timezone.utc,
    )
    return resolved.isoformat()


def clean_location_text(value):
    value = normalize_space(value)
    value = re.sub(r"\bSave this event:.*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bShare this event:.*$", "", value, flags=re.IGNORECASE)
    return normalize_space(value)


def parse_eventbrite_detail_metadata(event_url):
    response = safe_get(event_url, timeout=8, retries=1)
    if not response:
        return {"start_datetime_raw": "", "location": "", "description": ""}

    soup = BeautifulSoup(response.content, "html.parser")
    for script in soup.find_all("script", type="application/ld+json"):
        script_text = normalize_space(script.string or script.get_text(" ", strip=True))
        if not script_text:
            continue
        try:
            payload = json.loads(script_text)
        except json.JSONDecodeError:
            continue

        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            event_type = normalize_space(item.get("@type", ""))
            if event_type.lower() != "event":
                continue

            location = ""
            location_data = item.get("location")
            if isinstance(location_data, dict):
                location = normalize_space(location_data.get("name", ""))

            return {
                "start_datetime_raw": normalize_space(item.get("startDate", "")),
                "location": location,
                "description": normalize_space(item.get("description", "")),
            }

    return {"start_datetime_raw": "", "location": "", "description": ""}


def make_activity_id(title, start_datetime_raw, location):
    raw_key = "|".join(
        [
            normalize_space(title).lower(),
            normalize_space(start_datetime_raw).lower(),
            normalize_space(location).lower(),
        ]
    )
    return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:16]


def normalize_event(event):
    title = normalize_space(event.get("title", ""))
    start_datetime_raw = normalize_space(event.get("start_datetime_raw", ""))
    location = normalize_space(event.get("location", "Singapore")) or "Singapore"
    description = normalize_space(event.get("description", ""))
    url = normalize_space(event.get("url", ""))
    source = normalize_space(event.get("source", "unknown"))

    if not title:
        return None

    event_id = make_activity_id(title, start_datetime_raw, location)
    start_datetime_iso = parse_to_iso_datetime(start_datetime_raw)
    status = "upcoming" if start_datetime_iso else "undated"

    return {
        "activity_id": event_id,
        "title": title,
        "start_datetime_raw": start_datetime_raw,
        "start_datetime_iso": start_datetime_iso,
        "location": location,
        "description": description,
        "url": url,
        "source": source,
        "source_confidence": event.get("source_confidence", "medium"),
        "status": status,
        "last_seen_at": now_iso(),
    }


def parse_eventbrite(url):
    response = safe_get(url)
    if not response:
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    events = []
    seen_urls = set()
    detail_fetch_limit = 4
    detail_fetch_count = 0

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/e/" not in href:
            continue

        event_url = href if href.startswith("http") else urljoin(url, href)
        if event_url in seen_urls:
            continue

        seen_urls.add(event_url)
        parent = link.find_parent(["article", "li", "div", "section"]) or link
        context = normalize_space(parent.get_text(" ", strip=True))

        title = normalize_space(link.get_text(" ", strip=True))
        title = re.sub(r"^View\s+", "", title, flags=re.IGNORECASE)
        if not title:
            title = text_from_first(parent, ["h3", "h2"], default="Untitled Activity")

        date_text = extract_date_text(context)
        location = "Singapore"
        location_match = re.search(r"Singapore\s*[·\-]\s*([^\|]+)", context, flags=re.IGNORECASE)
        if location_match:
            location = clean_location_text(location_match.group(1))
        elif "Singapore" in context:
            location = "Singapore"

        detail_meta = {"start_datetime_raw": "", "location": "", "description": ""}
        if (not date_text or location == "Singapore") and detail_fetch_count < detail_fetch_limit:
            detail_meta = parse_eventbrite_detail_metadata(event_url)
            detail_fetch_count += 1

        if not date_text and detail_meta["start_datetime_raw"]:
            date_text = detail_meta["start_datetime_raw"]

        if location == "Singapore" and detail_meta["location"]:
            location = clean_location_text(detail_meta["location"])

        description = "Eventbrite listing for seniors-related activities"
        if detail_meta["description"]:
            description = detail_meta["description"][:280]

        events.append(
            {
                "title": title,
                "start_datetime_raw": date_text,
                "location": location,
                "description": description,
                "url": event_url,
                "source": "eventbrite",
                "source_confidence": "high" if date_text else "medium",
            }
        )

    return events


def parse_agewellsg(url):
    response = safe_get(url)
    if not response:
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    events = []
    links = soup.find_all("a", href=True)

    for link in links:
        text = normalize_space(link.get_text(" ", strip=True))
        href = link["href"]
        if not text:
            continue
        if "aac" not in text.lower() and "active ageing" not in text.lower():
            continue

        events.append(
            {
                "title": text,
                "start_datetime_raw": "",
                "location": "Singapore",
                "description": "Age Well SG active ageing programme/location reference",
                "url": href if href.startswith("http") else urljoin(url, href),
                "source": "agewellsg",
                "source_confidence": "low",
            }
        )

    return events


def parse_lionsbefrienders(url):
    response = safe_get(url)
    if not response:
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    events = []
    links = soup.find_all("a", href=True)

    for link in links:
        href = link["href"]
        title = normalize_space(link.get_text(" ", strip=True))
        if not title:
            continue
        lower_href = href.lower()
        if not any(keyword in lower_href for keyword in ["event", "campaign", "active-ageing", "stories"]):
            continue

        parent_text = normalize_space((link.find_parent(["article", "div", "li", "section"]) or link).get_text(" ", strip=True))
        date_text = extract_date_text(parent_text)

        events.append(
            {
                "title": title,
                "start_datetime_raw": date_text,
                "location": "Singapore",
                "description": "Lions Befrienders programme/news entry",
                "url": href if href.startswith("http") else urljoin(url, href),
                "source": "lionsbefrienders",
                "source_confidence": "medium" if date_text else "low",
            }
        )

    return events


def parse_onepa(url):
    response = safe_get(url)
    if not response:
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    events = []
    seen_urls = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/events/" not in href or any(skip in href for skip in ["/events/search", "/events?", "/events#"]):
            continue

        event_url = href if href.startswith("http") else urljoin(url, href)
        if event_url in seen_urls:
            continue
        seen_urls.add(event_url)

        title = normalize_space(link.get_text(" ", strip=True))
        title = re.sub(r"^EVENT\s+", "", title, flags=re.IGNORECASE)
        if not title:
            parent = link.find_parent(["article", "li", "div", "section"]) or link
            context = normalize_space(parent.get_text(" ", strip=True))
            title_match = re.search(r"EVENT\s+(.*?)\s+Ref\s*Code", context, flags=re.IGNORECASE)
            if title_match:
                title = normalize_space(title_match.group(1))
        if not title:
            continue

        parent = link.find_parent(["article", "li", "div", "section"]) or link
        context = normalize_space(parent.get_text(" ", strip=True))
        date_text = extract_date_text(context)

        location = "Singapore"
        location_match = re.search(
            r"Ref\s*Code\s*:\s*\d+\s*(.*?)\s*\d{1,2}\s+\w+\s+\d{4}",
            context,
            flags=re.IGNORECASE,
        )
        if location_match:
            candidate = normalize_space(location_match.group(1))
            if candidate and len(candidate) < 80:
                location = candidate

        events.append(
            {
                "title": title,
                "start_datetime_raw": date_text,
                "location": location,
                "description": "onePA Active Ageing event listing",
                "url": event_url,
                "source": "onepa",
                "source_confidence": "high" if date_text else "medium",
            }
        )

    return events


def parse_meetup(url):
    response = safe_get(url)
    if not response:
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    events = []
    seen_urls = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/events/" not in href:
            continue

        event_url = href if href.startswith("http") else urljoin(url, href)
        if event_url in seen_urls:
            continue
        seen_urls.add(event_url)

        title = normalize_space(link.get_text(" ", strip=True))
        if not title or len(title) < 6:
            continue

        parent = link.find_parent(["article", "li", "div", "section"]) or link
        context = normalize_space(parent.get_text(" ", strip=True))
        date_text = extract_date_text(context)
        if not date_text:
            alt_match = re.search(
                r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\w+\s+\d{1,2}\s+·\s+\d{1,2}:\d{2}\s+[AP]M\s+SST\b",
                context,
                flags=re.IGNORECASE,
            )
            if alt_match:
                date_text = normalize_space(alt_match.group(0))

        events.append(
            {
                "title": title,
                "start_datetime_raw": date_text,
                "location": "Singapore",
                "description": "Meetup seniors-related event listing",
                "url": event_url,
                "source": "meetup",
                "source_confidence": "medium" if date_text else "low",
            }
        )

    return events


def parse_visitsingapore(url):
    response = safe_get(url)
    if not response:
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    events = []
    seen_urls = set()

    for script in soup.find_all("script", type="application/ld+json"):
        script_text = normalize_space(script.string or script.get_text(" ", strip=True))
        if not script_text:
            continue
        try:
            payload = json.loads(script_text)
        except json.JSONDecodeError:
            continue

        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = normalize_space(item.get("@type", "")).lower()
            if item_type != "event":
                continue

            title = normalize_space(item.get("name", ""))
            event_url = normalize_space(item.get("url", ""))
            date_text = normalize_space(item.get("startDate", ""))
            if not title or not event_url or event_url in seen_urls:
                continue

            seen_urls.add(event_url)
            events.append(
                {
                    "title": title,
                    "start_datetime_raw": date_text,
                    "location": "Singapore",
                    "description": "VisitSingapore happenings listing",
                    "url": event_url,
                    "source": "visitsingapore",
                    "source_confidence": "high" if date_text else "medium",
                }
            )

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if any(skip in href.lower() for skip in ["privacy", "terms", "cookie", "sitemap", "contact-us"]):
            continue

        if href.startswith("http"):
            host = urlparse(href).netloc.lower()
            allowed_hosts = {
                "www.eventbrite.sg",
                "www.eventbrite.com",
                "www.gardensbythebay.com.sg",
                "www.nationalgallery.sg",
                "www.sentosa.com.sg",
                "www.nhb.gov.sg",
                "nus.edu",
                "www.ticketing.emint.com",
            }
            if host not in allowed_hosts and "visitsingapore.com" in host:
                continue

        parent = link.find_parent(["article", "li", "div", "section"]) or link
        context = normalize_space(parent.get_text(" ", strip=True))
        date_text = extract_date_text(context)
        if not date_text:
            range_match = re.search(
                r"\b\d{1,2}\s+\w+\s*[’']?\d{2}\s*-\s*\d{1,2}\s+\w+\s*[’']?\d{2}\b",
                context,
                flags=re.IGNORECASE,
            )
            if range_match:
                date_text = normalize_space(range_match.group(0))

        if not date_text:
            continue

        title = normalize_space(link.get_text(" ", strip=True))
        if not title or len(title) < 5:
            continue

        event_url = href if href.startswith("http") else urljoin(url, href)
        if event_url in seen_urls:
            continue
        seen_urls.add(event_url)

        events.append(
            {
                "title": title,
                "start_datetime_raw": date_text,
                "location": "Singapore",
                "description": "VisitSingapore happenings listing",
                "url": event_url,
                "source": "visitsingapore",
                "source_confidence": "medium",
            }
        )

    return events


def parse_timeoutsg(url):
    response = safe_get(url)
    if not response:
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    events = []
    seen_urls = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/singapore/" not in href:
            continue
        if any(skip in href for skip in ["/about", "/jobs", "/newsletter", "/sitemaps", "/privacy", "/terms"]):
            continue

        event_url = href if href.startswith("http") else urljoin(url, href)
        if event_url in seen_urls:
            continue

        title = normalize_space(link.get_text(" ", strip=True))
        if not title or len(title) < 10:
            continue

        parent = link.find_parent(["article", "li", "div", "section"]) or link
        context = normalize_space(parent.get_text(" ", strip=True))
        date_text = extract_date_text(context)

        seen_urls.add(event_url)
        events.append(
            {
                "title": title,
                "start_datetime_raw": date_text,
                "location": "Singapore",
                "description": "Time Out Singapore things-to-do/event listing",
                "url": event_url,
                "source": "timeoutsg",
                "source_confidence": "medium" if date_text else "low",
            }
        )

    return events


PARSERS = {
    "parse_eventbrite": parse_eventbrite,
    "parse_agewellsg": parse_agewellsg,
    "parse_lionsbefrienders": parse_lionsbefrienders,
    "parse_onepa": parse_onepa,
    "parse_meetup": parse_meetup,
    "parse_visitsingapore": parse_visitsingapore,
    "parse_timeoutsg": parse_timeoutsg,
}


def ingest_all_sources():
    all_events = []
    source_stats = {}

    for source in SOURCES:
        parser_name = source["parser"]
        parser_func = PARSERS[parser_name]
        raw_events = parser_func(source["url"])

        normalized = []
        for event in raw_events:
            normalized_event = normalize_event(event)
            if normalized_event:
                normalized.append(normalized_event)

        source_stats[source["name"]] = {
            "raw_count": len(raw_events),
            "normalized_count": len(normalized),
        }
        all_events.extend(normalized)

    deduped = dedupe_events(all_events)
    return deduped, source_stats


def dedupe_events(events):
    deduped = {}
    for event in events:
        event_id = event["activity_id"]
        existing = deduped.get(event_id)
        if not existing:
            deduped[event_id] = event
            continue

        if existing.get("source_confidence") == "low" and event.get("source_confidence") in {"medium", "high"}:
            deduped[event_id] = event

    return list(deduped.values())


def events_to_dataframe(events):
    df = pd.DataFrame(events)
    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[EXPECTED_COLUMNS]


def build_rag_record(row):
    title = normalize_space(row.get("title", ""))
    date_text = normalize_space(row.get("start_datetime_raw", ""))
    location = normalize_space(row.get("location", ""))
    description = normalize_space(row.get("description", ""))
    source = normalize_space(row.get("source", ""))
    url = normalize_space(row.get("url", ""))

    categories = classify_categories_llm(title, description, source, location)
    category_source = "llm"
    if not categories:
        categories = classify_categories_keyword(title, description, source, location)
        category_source = "keyword"

    rewritten_summary = build_deterministic_summary(title, date_text, location, description)
    rewrite_source = "deterministic"

    category_csv = ", ".join(categories)

    content = "\n".join(
        [
            f"Title: {title}" if title else "",
            f"Date: {date_text}" if date_text else "",
            f"Location: {location}" if location else "",
            f"Senior Summary: {rewritten_summary}" if rewritten_summary else "",
            f"Description: {description}" if description else "",
            f"Source: {source}" if source else "",
            f"Link: {url}" if url else "",
            f"Audience: Seniors" if title or description else "",
            f"Categories: {category_csv}" if category_csv else "",
            f"Category Source: {category_source}",
            f"Rewrite Source: {rewrite_source}",
            "Keywords: senior-friendly, singapore activities",
        ]
    ).strip()

    raw_bytes = base64.b64encode(content.encode("utf-8")).decode("ascii")

    return {
        "id": row.get("activity_id", ""),
        "structData": {
            "title": title,
            "source": source,
            "location": location,
            "uri": url,
            "audience": "seniors",
            "categories": categories,
            "categories_csv": category_csv,
            "category_source": category_source,
            "senior_summary": rewritten_summary,
            "rewrite_source": rewrite_source,
            "status": row.get("status", ""),
            "start_datetime_raw": date_text,
            "start_datetime_iso": row.get("start_datetime_iso", ""),
            "source_confidence": row.get("source_confidence", ""),
            "last_seen_at": row.get("last_seen_at", ""),
        },
        "content": {
            "mimeType": "text/plain",
            "rawBytes": raw_bytes,
        },
    }


def save_jsonl_for_rag(df):
    with open(DATASTORE_JSONL, "w", encoding="utf-8") as jsonl_file:
        for row in df.to_dict(orient="records"):
            rag_record = build_rag_record(row)
            jsonl_file.write(json.dumps(rag_record, ensure_ascii=False) + "\n")


def save_datastore(events, source_stats):
    ensure_output_dir()
    df = events_to_dataframe(events)

    if not df.empty:
        df = df.sort_values(by=["start_datetime_iso", "title"], na_position="last").reset_index(drop=True)

    json_records = df.to_dict(orient="records")
    with open(DATASTORE_JSON, "w", encoding="utf-8") as json_file:
        json.dump(json_records, json_file, indent=2, ensure_ascii=False)

    df.to_csv(DATASTORE_CSV, index=False)
    save_jsonl_for_rag(df)

    upload_to_gcs(DATASTORE_JSON, "activities_latest.json")
    upload_to_gcs(DATASTORE_CSV, "activities_latest.csv")
    upload_to_gcs(DATASTORE_JSONL, "activities_latest_rag.jsonl")

    report = {
        "generated_at": now_iso(),
        "total_records": len(df),
        "dated_records": int((df["status"] == "upcoming").sum()) if not df.empty else 0,
        "undated_records": int((df["status"] == "undated").sum()) if not df.empty else 0,
        "tagging": {
            "llm_enabled": ENABLE_LLM_TAGGING,
            "llm_disabled_reason": _llm_tag_disabled_reason,
            "llm_project": VERTEX_PROJECT_ID,
            "llm_location": VERTEX_LOCATION,
            "llm_model": VERTEX_MODEL,
            "llm_calls_made": _llm_calls_made,
            "llm_cache_size": len(_llm_category_cache),
        },
        "rewrite": {
            "mode": "deterministic",
            "llm_enabled": False,
        },
        "sources": source_stats,
        "outputs": {
            "json": DATASTORE_JSON,
            "csv": DATASTORE_CSV,
            "jsonl": DATASTORE_JSONL,
        },
    }

    with open(RUN_REPORT_JSON, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)

    upload_to_gcs(RUN_REPORT_JSON, "ingestion_report.json")

    return df, report


def recommend_activities(user_interest, user_location, events_df=None):
    df = events_df if events_df is not None else events_to_dataframe([])
    if df.empty:
        return df

    title_series = df["title"].fillna("").astype(str)
    description_series = df["description"].fillna("").astype(str)
    location_series = df["location"].fillna("").astype(str)

    interest_match = title_series.str.contains(user_interest, case=False, regex=False) | description_series.str.contains(
        user_interest,
        case=False,
        regex=False,
    )
    location_match = location_series.str.contains(user_location, case=False, regex=False)

    return df[interest_match & location_match]


def run_ingestion():
    events, source_stats = ingest_all_sources()
    df, report = save_datastore(events, source_stats)

    print("Data ingestion completed.")
    print(f"Total records: {report['total_records']}")
    print(f"Dated records: {report['dated_records']}")
    print(f"Undated records: {report['undated_records']}")
    print(f"Saved JSON: {DATASTORE_JSON}")
    print(f"Saved CSV: {DATASTORE_CSV}")
    print(f"Saved JSONL (RAG): {DATASTORE_JSONL}")
    print(f"Run report: {RUN_REPORT_JSON}")
    if BUCKET_NAME:
        print(f"Uploaded artifacts to bucket: gs://{BUCKET_NAME}/")

    return df


def run_job():
    if IS_CLOUD_RUNTIME and not BUCKET_NAME:
        raise RuntimeError(
            "Missing GCS_BUCKET_NAME environment variable. "
            "Set it to your target bucket before running this Cloud Run Job."
        )

    dataframe = run_ingestion()
    print(f"Job completed. Processed {len(dataframe)} activities.")
    return dataframe


if __name__ == "__main__":
    try:
        dataframe = run_job()
    except Exception as exc:
        print(f"[ERROR] Job failed: {exc}")
        raise SystemExit(1)

    if not IS_CLOUD_RUNTIME:
        user_interest = "fitness"
        user_location = "Singapore"
        recommendations = recommend_activities(user_interest, user_location, dataframe)

        print("\nSample recommendations (fitness + Singapore):")
        if recommendations.empty:
            print("No matching activities in current dataset.")
        else:
            print(recommendations[["title", "start_datetime_raw", "location", "source"]].head(10))

    raise SystemExit(0)