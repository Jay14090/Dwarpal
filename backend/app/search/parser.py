"""Natural-language query -> strict search filter (CLAUDE.md section 4, P7).

Filter: {entity, roles[], upper_color, lower_color, height_cm{min,max}, cameras[], zones[],
time_range{from,to}, plate, free_text}. Two parsers share the schema:

- RuleParser: deterministic regex/keyword parser, always available (no API key needed).
- LlmParser: provider-agnostic. `anthropic` uses the official SDK with structured outputs;
  `openai` speaks the OpenAI-compatible chat API (OpenAI, Groq, Together, Ollama, ...) in JSON mode.
  Output is validated against the same schema; any failure falls back to RuleParser.
Only the query text (plus camera/zone names and the current time) is sent to an LLM.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

from app.anpr.normalize import normalize
from app.core.config import CamerasConfig, LLMSettings
from app.pipeline.attributes import COLORS

log = logging.getLogger(__name__)

Color = Literal[
    "black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"
]
Role = Literal["resident", "staff", "unknown"]


class HeightRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min: float | None = None
    max: float | None = None


class TimeRange(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    from_: datetime | None = Field(None, alias="from")
    to: datetime | None = None


class SearchFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity: Literal["person", "vehicle"] = "person"
    roles: list[Role] = Field(default_factory=list)
    upper_color: Color | None = None
    lower_color: Color | None = None
    height_cm: HeightRange | None = None
    cameras: list[str] = Field(default_factory=list)
    zones: list[str] = Field(default_factory=list)
    time_range: TimeRange | None = None
    plate: str | None = None
    free_text: str | None = None

    def dump(self) -> dict:
        return self.model_dump(mode="json", by_alias=True)


# --------------------------------------------------------------------------- vocabulary

COLOR_WORDS = {c: c for c in COLORS} | {
    "grey": "gray", "silver": "gray", "maroon": "red", "crimson": "red", "navy": "blue", "violet": "purple",
    "beige": "brown", "khaki": "brown", "tan": "brown", "golden": "yellow", "olive": "green", "cream": "white",
    "saffron": "orange", "magenta": "pink", "dark": None,  # "dark" alone is not a colour
}  # fmt: skip
UPPER_WORDS = {"shirt", "t-shirt", "tshirt", "tee", "top", "jacket", "hoodie", "sweater", "sweatshirt", "kurta",
               "coat", "blouse", "vest", "uniform", "jersey", "polo", "blazer", "cardigan", "kurti"}  # fmt: skip
LOWER_WORDS = {"pants", "trousers", "jeans", "shorts", "skirt", "leggings", "pyjama", "pajama", "lowers",
               "trackpants", "track-pants", "dhoti", "salwar", "chinos"}  # fmt: skip
ROLE_WORDS = {
    "resident": "resident", "residents": "resident", "owner": "resident", "owners": "resident",
    "staff": "staff", "guard": "staff", "guards": "staff", "security": "staff", "maid": "staff",
    "housekeeping": "staff", "cleaner": "staff", "employee": "staff", "employees": "staff", "worker": "staff",
    "unknown": "unknown", "stranger": "unknown", "strangers": "unknown", "visitor": "unknown",
    "visitors": "unknown", "outsider": "unknown", "outsiders": "unknown", "unidentified": "unknown",
    "intruder": "unknown", "intruders": "unknown", "unregistered": "unknown",
}  # fmt: skip
VEHICLE_WORDS = {"car", "cars", "vehicle", "vehicles", "bike", "bikes", "motorcycle", "motorbike", "scooter",
                 "truck", "trucks", "bus", "auto", "rickshaw", "taxi", "cab", "plate", "number", "registration"}  # fmt: skip
STOP = {"a", "an", "the", "in", "on", "at", "of", "with", "wearing", "who", "was", "were", "is", "are", "and",
        "or", "show", "me", "find", "all", "any", "people", "person", "persons", "someone", "somebody", "near",
        "around", "by", "today", "yesterday", "tonight", "after", "before", "between", "since", "from", "to",
        "am", "pm", "last", "past", "this", "that", "did", "when", "where", "enter", "entered", "leave", "left",
        "come", "came", "seen", "spotted", "about", "approximately", "roughly", "hour", "hours", "minutes",
        "morning", "afternoon", "evening", "night", "foot", "feet", "ft", "cm", "tall", "short", "height",
        "camera", "cameras", "zone", "area", "please", "list", "everyone", "anyone", "there", "here", "dressed"}  # fmt: skip
PLATE_RE = re.compile(
    r"\b(\d{2}\s?BH\s?\d{4}\s?[A-Z]{1,2}|[A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{0,3}[\s-]?\d{4})\b", re.I
)
PERSON_WORDS = {"guy", "man", "woman", "lady", "boy", "girl", "kid", "child"}
NUM_WORDS = {"four": 4, "five": 5, "six": 6, "seven": 7, "three": 3, "one": 1, "two": 2, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12}  # fmt: skip


def _num(tok: str) -> float | None:
    if tok.replace(".", "", 1).isdigit():
        return float(tok)
    return float(NUM_WORDS[tok]) if tok in NUM_WORDS else None


class RuleParser:
    def __init__(self, cameras: CamerasConfig, tz: str = "Asia/Kolkata") -> None:
        self.cameras = cameras
        self.tz = ZoneInfo(tz)
        self.zone_names = sorted({z.name for c in cameras.cameras for z in c.zones})
        self.camera_zones = {c.id: [z.name for z in c.zones] for c in cameras.cameras}
        # zone keyword -> zone names (e.g. "parking" -> ["resident_parking", "visitor_parking"])
        self.zone_words: dict[str, set[str]] = {}
        for cam in cameras.cameras:
            for z in cam.zones:
                for w in re.split(r"[_\s-]+", z.name.lower()):
                    if len(w) > 2 and w not in ROLE_WORDS and w not in {"floor", "zone", "area"}:
                        self.zone_words.setdefault(w, set()).add(z.name)

    # ------------------------------------------------------------------ pieces

    def _times(self, q: str, now: datetime) -> tuple[TimeRange | None, str]:
        day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if re.search(r"\byesterday\b", q):
            day = day - timedelta(days=1)
            base = TimeRange(from_=day, to=day + timedelta(days=1))
            q = re.sub(r"\byesterday\b", " ", q)
        elif re.search(r"\b(today|tonight)\b", q):
            base = TimeRange(from_=day, to=now)
            q = re.sub(r"\b(today|tonight)\b", " ", q)
        else:
            base = None

        def at(h: int, m: int) -> datetime:
            return day.replace(hour=h % 24, minute=m)

        def hhmm(hs: str, ms: str | None, ap: str | None) -> tuple[int, int]:
            h, m = int(hs), int(ms or 0)
            if ap == "pm" and h < 12:
                h += 12
            if ap == "am" and h == 12:
                h = 0
            return h, m

        clock = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
        tr = None
        if m := re.search(rf"\bbetween\s+{clock}\s+(?:and|to|-)\s+{clock}", q):
            ap2 = m[6]
            h1, m1 = hhmm(m[1], m[2], m[3] or ap2)
            h2, m2 = hhmm(m[4], m[5], ap2)
            tr = TimeRange(from_=at(h1, m1), to=at(h2, m2))
        elif m := re.search(rf"\b(after|since|from)\s+{clock}", q):
            h, mi = hhmm(m[2], m[3], m[4] or ("pm" if int(m[2]) < 8 else None))
            end = (base.to if base else None) or (day + timedelta(days=1))
            tr = TimeRange(from_=at(h, mi), to=end)
        elif m := re.search(rf"\bbefore\s+{clock}", q):
            h, mi = hhmm(m[1], m[2], m[3])
            tr = TimeRange(from_=day, to=at(h, mi))
        elif m := re.search(rf"\b(?:around|at|about)\s+{clock}\b", q):
            if m[3] or m[2]:
                h, mi = hhmm(m[1], m[2], m[3])
                c = at(h, mi)
                tr = TimeRange(from_=c - timedelta(minutes=30), to=c + timedelta(minutes=30))
        elif m := re.search(
            r"\b(?:last|past)\s+(\d+|an?|one|two|three)?\s*(hour|hours|minute|minutes|min|mins)\b",
            q,
        ):
            n = (
                1
                if m[1] in (None, "a", "an", "one")
                else (int(m[1]) if m[1].isdigit() else NUM_WORDS[m[1]])
            )
            delta = timedelta(hours=n) if m[2].startswith("hour") else timedelta(minutes=n)
            tr = TimeRange(from_=now - delta, to=now)
        else:
            parts = {
                "morning": (6, 12),
                "afternoon": (12, 17),
                "evening": (17, 21),
                "night": (21, 30),
            }
            for word, (a, b) in parts.items():
                if re.search(rf"\b(this\s+)?{word}\b", q):
                    tr = TimeRange(from_=at(a, 0), to=day + timedelta(hours=b))
                    break
        if tr is not None:
            q = re.sub(
                rf"\b(between|after|since|from|before|around|at|about)\s+{clock}(\s+(and|to|-)\s+{clock})?",
                " ",
                q,
            )
        return tr or base, q

    def _height(self, q: str) -> tuple[HeightRange | None, str]:
        # 5'8", 5 ft 8, five foot eight, six foot, 180 cm, tall/short
        if m := re.search(r"\b(\d{3})\s*cm\b", q):
            c = float(m[1])
            return HeightRange(min=c - 5, max=c + 5), q.replace(m[0], " ")
        pat = r"\b(\d|four|five|six|seven)\s*(?:'|ft|feet|foot)(?:\s*(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven)\s*(?:\"|in|inch|inches)?)?"
        if m := re.search(pat, q):
            ft, inch = _num(m[1]) or 0, _num(m[2]) if m[2] else None
            if inch is None:  # "six foot" ~ 6'0"-6'2"+ -> CLAUDE.md: about 175-190 cm
                c = ft * 30.48
                return HeightRange(min=round(c - 8), max=round(c + 7)), q.replace(m[0], " ")
            c = ft * 30.48 + inch * 2.54
            return HeightRange(min=round(c - 5), max=round(c + 5)), q.replace(m[0], " ")
        if re.search(r"\btall\b", q):
            return HeightRange(min=180, max=None), re.sub(r"\btall\b", " ", q)
        if re.search(r"\bshort\b", q):
            return HeightRange(min=None, max=162), re.sub(r"\bshort\b", " ", q)
        return None, q

    def _colors(self, tokens: list[str]) -> tuple[str | None, str | None]:
        upper = lower = None
        loose: list[tuple[int, str]] = []
        for i, tok in enumerate(tokens):
            color = COLOR_WORDS.get(tok)
            if color is None:
                continue
            target = None
            for j in range(i + 1, min(i + 4, len(tokens))):
                if tokens[j] in UPPER_WORDS:
                    target = "upper"
                    break
                if tokens[j] in LOWER_WORDS:
                    target = "lower"
                    break
            if target is None:  # "shirt in red"
                for j in range(max(0, i - 3), i):
                    if tokens[j] in UPPER_WORDS:
                        target = "upper"
                    elif tokens[j] in LOWER_WORDS:
                        target = "lower"
            if target == "upper" and upper is None:
                upper = color
            elif target == "lower" and lower is None:
                lower = color
            elif target is None:
                loose.append((i, color))
        for _, color in loose:  # "in red" -> upper, "red and black" -> upper, lower
            if upper is None:
                upper = color
            elif lower is None:
                lower = color
        return upper, lower

    # ------------------------------------------------------------------ main

    def parse(self, query: str, now: datetime | None = None) -> SearchFilter:
        now = (now or datetime.now(self.tz)).astimezone(self.tz)
        q = " " + query.lower().replace("\u2019", "'") + " "
        f = SearchFilter()

        if m := PLATE_RE.search(query):
            p = normalize(m[1])
            if p.valid:
                f.plate = p.text
                f.entity = "vehicle"
                q = q.replace(m[1].lower(), " ")
        f.time_range, q = self._times(q, now)
        f.height_cm, q = self._height(q)

        # whole camera names / ids and zone names first ("resident parking" is a place, not a role)
        zones: set[str] = set()
        phrases: list[str] = []
        for cam in self.cameras.cameras:
            for key in (cam.id, cam.name.lower()):
                if re.search(rf"\b{re.escape(key)}\b", q):
                    if cam.id not in f.cameras:
                        f.cameras.append(cam.id)
                    phrases.append(key)
        for z in self.zone_names:
            phrase = z.replace("_", " ")
            if " " in phrase and re.search(rf"\b{re.escape(phrase)}\b", q):
                zones.add(z)
                phrases.append(phrase)
        for phrase in sorted(phrases, key=len, reverse=True):
            q = re.sub(rf"\b{re.escape(phrase)}\b", " ", q)

        tokens = re.findall(r"[a-z][a-z'\-]*|\d+", q)
        if any(t in VEHICLE_WORDS for t in tokens):
            f.entity = "vehicle"
        f.roles = sorted({ROLE_WORDS[t] for t in tokens if t in ROLE_WORDS})  # type: ignore[misc]
        if f.entity == "person":
            f.upper_color, f.lower_color = self._colors(tokens)  # type: ignore[assignment]
        for t in tokens:
            zones |= self.zone_words.get(t, set()) | self.zone_words.get(t.rstrip("s"), set())
        f.zones = sorted(zones)

        # free text: words the structured fields do not capture (garments, objects, ...)
        handled = (
            set(ROLE_WORDS)
            | VEHICLE_WORDS
            | STOP
            | set(self.zone_words)
            | {w + "s" for w in self.zone_words}
        )
        keep = [t for t in tokens if t not in handled and not t.isdigit()]
        if f.entity == "person" and set(keep) - PERSON_WORDS:
            f.free_text = " ".join(keep)
        return f


# --------------------------------------------------------------------------- LLM

SYSTEM_PROMPT = """You convert a security operator's search request about CCTV footage of a gated residential community into a JSON search filter.

Fields:
- entity: "vehicle" for cars, bikes, plates or registration numbers; otherwise "person".
- roles: any of "resident", "staff", "unknown" (strangers, visitors, intruders are "unknown"). Empty when not mentioned.
- upper_color / lower_color: clothing colours, one of black, white, gray, red, orange, yellow, green, blue, purple, pink, brown. Upper is the torso (shirt, t-shirt, jacket, kurta); lower is the legs (pants, jeans, shorts, skirt). A colour without a garment is upper. Null when not mentioned.
- height_cm: {"min", "max"} in centimetres. "Six foot" is about 175-190; "tall" means min 180; "short" means max 162. Null when not mentioned.
- cameras: camera ids from the list below that the request names. zones: zone names from the list below that the request refers to (e.g. "near the gate" matches zones containing "gate").
- time_range: {"from", "to"} as ISO 8601 datetimes with the timezone offset, resolved against the current time below. "After 9 pm" means from 21:00 today to midnight; "today" means from 00:00 to now. Null when no time is mentioned.
- plate: an Indian registration number in canonical form without spaces (e.g. TN09AB1234, 22BH1234AB), else null.
- free_text: a short visual description for image-text matching (e.g. "man in a green t-shirt with a backpack"), or null when the request has no visual details beyond the structured fields.
Return only the JSON object."""


class LlmParser:
    def __init__(self, cfg: LLMSettings, cameras: CamerasConfig, tz: str = "Asia/Kolkata") -> None:
        self.cfg = cfg
        self.tz = ZoneInfo(tz)
        self.context = json.dumps(
            {
                "cameras": [
                    {"id": c.id, "name": c.name, "zones": [z.name for z in c.zones]}
                    for c in cameras.cameras
                ]
            }
        )

    def _user(self, query: str, now: datetime) -> str:
        return f"Current time: {now.isoformat()}\nCameras and zones: {self.context}\n\nRequest: {query}"

    def parse(self, query: str, now: datetime | None = None) -> SearchFilter:
        now = (now or datetime.now(self.tz)).astimezone(self.tz)
        provider = self.cfg.provider.lower()
        if provider == "anthropic":
            return self._anthropic(query, now)
        if provider in ("openai", "openai-compatible", "groq", "together", "ollama"):
            return self._openai_compatible(query, now)
        raise ValueError(
            f"unsupported LLM_PROVIDER {self.cfg.provider!r} (use anthropic or openai)"
        )

    def _anthropic(self, query: str, now: datetime) -> SearchFilter:
        import anthropic

        client = anthropic.Anthropic(
            api_key=self.cfg.api_key or None, timeout=self.cfg.timeout_s, max_retries=1
        )
        response = client.messages.parse(
            model=self.cfg.model or "claude-opus-5-5",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            output_config={"effort": self.cfg.effort},
            messages=[{"role": "user", "content": self._user(query, now)}],
            output_format=SearchFilter,
        )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise ValueError(f"no parsed output (stop_reason={response.stop_reason})")
        return response.parsed_output

    def _openai_compatible(self, query: str, now: datetime) -> SearchFilter:
        import httpx2

        base = (self.cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        schema = json.dumps(SearchFilter.model_json_schema(by_alias=True))
        body = {
            "model": self.cfg.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT + "\nJSON schema: " + schema},
                {"role": "user", "content": self._user(query, now)},
            ],
        }
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else {}
        r = httpx2.post(
            f"{base}/chat/completions", json=body, headers=headers, timeout=self.cfg.timeout_s
        )
        r.raise_for_status()
        return SearchFilter.model_validate_json(r.json()["choices"][0]["message"]["content"])


class QueryParser:
    """LLM when configured, rules otherwise or on any LLM failure."""

    def __init__(self, llm: LLMSettings, cameras: CamerasConfig, tz: str = "Asia/Kolkata") -> None:
        self.rules = RuleParser(cameras, tz)
        self.llm = LlmParser(llm, cameras, tz) if llm.enabled else None

    def parse(self, query: str, now: datetime | None = None) -> tuple[SearchFilter, str]:
        if self.llm is not None:
            try:
                return self.llm.parse(query, now), f"llm:{self.llm.cfg.provider}"
            except Exception as exc:
                log.warning("LLM parse failed (%s: %s); using rules", type(exc).__name__, exc)
        return self.rules.parse(query, now), "rules"


def time_of_day(t: time) -> str:  # small helper for UI labels
    return t.strftime("%H:%M")
