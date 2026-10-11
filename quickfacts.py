"""Quick live facts the front desk can use on a call: weather (Open-Meteo, no key) and the local date and time.
Only the place name and coordinates go out; nothing about the owner's contacts. Returns '' when the question is not about these."""
import json
import urllib.parse
import urllib.request
from datetime import datetime

import config

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo(config.get("timezone"))
except Exception:  # pragma: no cover
    TZ = None

PLACES = {k: tuple(v) for k, v in config.get("places").items()}
DEFAULT_PLACE = config.get("default_place")

WEATHER_HINTS = ("weather", "temperature", "forecast", "rain", "snow", "umbrella", "how cold", "how hot", "how warm", "windy",
                 "\u5929\u6c14", "\u6e29\u5ea6", "\u4e0b\u96e8", "\u4e0b\u96ea", "\u9884\u62a5", "\u5e26\u4f1e")
TIME_HINTS = ("what time", "time is it", "what day", "what's the date", "whats the date", "today's date", "what date",
              "\u51e0\u70b9", "\u661f\u671f\u51e0", "\u51e0\u53f7", "\u65e5\u671f")

CODES = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy", 51: "light drizzle", 53: "drizzle",
         55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
         71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "rain showers", 81: "rain showers", 82: "heavy showers",
         85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail"}


def _get(url, timeout=4):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _place(text):
    t = text.lower()
    for name in PLACES:
        if name in t:
            return name
    return DEFAULT_PLACE


def weather_facts(text, get=_get):
    name = _place(text)
    lat, lon = PLACES[name]
    q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "timezone": config.get("timezone"), "forecast_days": 2,
                                "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"})
    d = get("https://api.open-meteo.com/v1/forecast?" + q)
    c, day = d["current"], d["daily"]
    now = "%s, %d degrees C (feels like %d), wind %d km/h" % (CODES.get(c.get("weather_code"), "unknown"), round(c["temperature_2m"]),
                                                            round(c["apparent_temperature"]), round(c["wind_speed_10m"]))
    def one(i, label):
        return "%s: %s, high %d, low %d, chance of precipitation %s%%" % (
            label, CODES.get(day["weather_code"][i], "unknown"), round(day["temperature_2m_max"][i]),
            round(day["temperature_2m_min"][i]), day["precipitation_probability_max"][i])
    return "Weather in %s now: %s. %s. %s." % (name.title(), now, one(0, "Today"), one(1, "Tomorrow"))


def time_facts(now=None):
    n = now or (datetime.now(TZ) if TZ else datetime.now())
    return "Local time in %s: %s." % (config.get("time_label"), n.strftime("%A, %B %d %Y, %I:%M %p"))


def quick_facts(text, get=_get, now=None):
    """LIVE FACTS block for the notes, or '' if nothing applies or a lookup fails."""
    t = text.lower()
    out = []
    try:
        if any(h in t for h in WEATHER_HINTS):
            out.append(weather_facts(text, get))
        if any(h in t for h in TIME_HINTS) or out:
            out.append(time_facts(now))
    except Exception:
        return ""
    return ("LIVE FACTS:\n" + "\n".join(out)) if out else ""
