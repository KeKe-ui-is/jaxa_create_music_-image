# app.py
import io
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import streamlit as st
from PIL import Image
from scipy.io import wavfile

import torch
from diffusers import StableDiffusionPipeline

# MusicGen (Transformers)
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# JAXA Earth SDK
from jaxa.earth import je

# Map click
import folium
from streamlit_folium import st_folium

import requests


# =========================
# Utils
# =========================
def normalize(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return float(max(0.0, min(1.0, (value - lower) / (upper - lower))))


def month_range_one_month_before(target_month: date) -> Tuple[str, str]:
    y, m = target_month.year, target_month.month
    if m == 1:
        y -= 1
        m = 12
    else:
        m -= 1
    start = date(y, m, 1)
    if m == 12:
        end = date(y, 12, 31)
    else:
        end = date(y, m + 1, 1) - timedelta(days=1)
    return f"{start.isoformat()}T00:00:00", f"{end.isoformat()}T23:59:59"


def week_range_one_week_before(target_month: date) -> Tuple[str, str]:
    end = target_month - timedelta(days=1)
    start = end - timedelta(days=6)
    return f"{start.isoformat()}T00:00:00", f"{end.isoformat()}T23:59:59"


def fallback_values(lat: float, lon: float) -> Dict[str, float]:
    seed = int(abs(lat) * 10000 + abs(lon) * 10000)
    rng = np.random.default_rng(seed)
    return {
        "ndvi": float(rng.uniform(0.1, 0.9)),
        "lst": float(rng.uniform(-5.0, 40.0)),
        "precip_week": float(rng.uniform(0.0, 120.0)),
        "precip_month": float(rng.uniform(10.0, 450.0)),
    }


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    d1 = math.radians(lat2 - lat1)
    d2 = math.radians(lon2 - lon1)
    a = math.sin(d1 / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d2 / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# =========================
# Reverse Geocoding + Urban/Rural classification
# =========================
MAJOR_CITIES = [
    ("Tokyo", 35.681236, 139.767125, 30),
    ("Yokohama", 35.443707, 139.638031, 25),
    ("Osaka", 34.693737, 135.502165, 25),
    ("Nagoya", 35.170915, 136.881537, 22),
    ("Sapporo", 43.061771, 141.354451, 22),
    ("Fukuoka", 33.590355, 130.401716, 18),
    ("Kobe", 34.690083, 135.195511, 18),
    ("Kyoto", 35.011564, 135.768149, 18),
    ("Sendai", 38.268215, 140.869356, 18),
    ("Hiroshima", 34.385203, 132.455293, 18),
]


def reverse_geocode_nominatim(lat: float, lon: float) -> Dict[str, Any]:
    """
    OpenStreetMap Nominatim を使って県/市区町村などを取得（ネット接続が必要）
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "format": "jsonv2",
        "lat": str(lat),
        "lon": str(lon),
        "zoom": "10",
        "addressdetails": "1",
        "accept-language": "ja,en",
    }
    headers = {
        "User-Agent": "jaxa-hackathon-demo/1.0 (streamlit app)",
    }
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()


def parse_pref_city(geo: Dict[str, Any]) -> Tuple[str, str, str]:
    addr = geo.get("address", {}) if isinstance(geo, dict) else {}
    pref = addr.get("state") or addr.get("province") or addr.get("region") or ""
    city = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("municipality")
        or addr.get("county")
        or addr.get("city_district")
        or ""
    )
    display = geo.get("display_name", "") if isinstance(geo, dict) else ""
    return pref, city, display


def classify_area(lat: float, lon: float, geo: Optional[Dict[str, Any]]) -> str:
    for _, clat, clon, radius_km in MAJOR_CITIES:
        if haversine_km(lat, lon, clat, clon) <= radius_km:
            return "urban"

    if not geo:
        return "rural"

    importance = float(geo.get("importance") or 0.0)
    addr = geo.get("address", {}) or {}
    cues_urban = {"city", "suburb", "neighbourhood", "commercial", "retail", "industrial"}
    cues_rural = {"village", "hamlet", "farmland", "forest", "mountain", "park"}

    addr_text = " ".join([str(v) for v in addr.values()]).lower()

    score = 0.0
    score += importance * 2.0

    for w in cues_urban:
        if w in addr_text:
            score += 0.35
    for w in cues_rural:
        if w in addr_text:
            score -= 0.25

    if score >= 1.1:
        return "urban"
    if score >= 0.7:
        return "suburban"
    return "rural"


# =========================
# JAXA Earth fetch
# =========================
@dataclass
class SatValues:
    ndvi: float
    lst: float
    precip_month: float
    precip_week_for_image: float
    used_month_range: Tuple[str, str]
    used_week_range: Tuple[str, str]
    source: str  # "jaxa" or "fallback" or "mixed"
    debug: Dict[str, str]


def _mean_from_timeseries(images) -> float:
    ip = je.ImageProcess(images).calc_spatial_stats()
    ts = getattr(ip, "timeseries", None)
    if ts is None:
        ss = getattr(ip, "spatial_stats", None)
        if ss is not None:
            if isinstance(ss, dict) and "mean" in ss:
                v = ss["mean"]
                arr = np.array(v, dtype=np.float32).reshape(-1)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    return float(arr.mean())
        raise TypeError("timeseries not found in ImageProcess result")

    if isinstance(ts, dict):
        for k in ("mean", "avg", "average"):
            if k in ts:
                v = np.array(ts[k], dtype=np.float32).reshape(-1)
                v = v[np.isfinite(v)]
                if v.size:
                    return float(v.mean())
        for _, v in ts.items():
            try:
                arr = np.array(v, dtype=np.float32).reshape(-1)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    return float(arr.mean())
            except Exception:
                continue

    if isinstance(ts, (list, tuple)):
        vals = []
        for it in ts:
            try:
                vals.append(float(it))
            except Exception:
                pass
        if vals:
            return float(np.mean(vals))

    raise TypeError(f"Could not extract mean from timeseries. type(timeseries)={type(ts)}")


def _pick_preferred_collection(candidates: List[Tuple[List[str], str]]) -> Optional[Tuple[str, str]]:
    for keywords, preferred_band in candidates:
        try:
            cols, bands = je.ImageCollectionList(ssl_verify=True).filter_name(keywords=keywords)
            if not cols:
                continue
            col = cols[0]
            band_list = bands[0] or []
            if preferred_band and preferred_band in band_list:
                return col, preferred_band
            if band_list:
                return col, band_list[0]
        except Exception:
            continue
    return None


def fetch_satellite_values(lat: float, lon: float, target_month: date, ppu: int = 40) -> SatValues:
    d = 0.15
    bbox = [lon - d, lat - d, lon + d, lat + d]

    month_start, month_end = month_range_one_month_before(target_month)
    week_start, week_end = week_range_one_week_before(target_month)

    debug: Dict[str, str] = {
        "month_range": f"{month_start} .. {month_end}",
        "week_range": f"{week_start} .. {week_end}",
        "bbox": str(bbox),
    }

    fb = fallback_values(lat, lon)

    NDVI_COLLECTION = "JAXA.G-Portal_GCOM-C.SGLI_standard.L3-NDVI.daytime.v3_global_monthly"
    NDVI_BAND = "NDVI"

    LST_COLLECTION_FALLBACK = "NASA.EOSDIS_Aqua.MODIS_MYD11C1-LST.daytime.v061_global_half-monthly-normal"
    LST_BAND_FALLBACK = "LST_2012_2021"
    lst_pick = _pick_preferred_collection(
        [
            (["MYD11C1", "LST", "half-monthly"], "LST_2012_2021"),
            (["LST", "half-monthly"], "LST_2012_2021"),
            (["LST", "monthly"], "LST"),
            (["LST"], "LST"),
        ]
    )
    if lst_pick is None:
        lst_pick = (LST_COLLECTION_FALLBACK, LST_BAND_FALLBACK)

    PRECIP_COLLECTION = "JAXA.EORC_GSMaP_standard.Gauge.00Z-23Z.v6_daily"
    PRECIP_BAND = "PRECIP"

    src = "jaxa"
    ndvi = fb["ndvi"]
    lst_c = fb["lst"]
    precip_week = fb["precip_week"]
    precip_month = fb["precip_month"]

    try:
        ndvi_imgs = (
            je.ImageCollection(collection=NDVI_COLLECTION, ssl_verify=True)
            .filter_date(dlim=[month_start, month_end])
            .filter_resolution(ppu=ppu)
            .filter_bounds(bbox=bbox)
            .select(band=NDVI_BAND)
            .get_images()
        )
        ndvi = _mean_from_timeseries(ndvi_imgs)
        debug["ndvi_collection"] = NDVI_COLLECTION
        debug["ndvi_band"] = NDVI_BAND
    except Exception as e:
        src = "mixed"
        debug["ndvi_error"] = str(e)

    try:
        LST_COLLECTION, LST_BAND = lst_pick
        debug["lst_collection"] = LST_COLLECTION
        debug["lst_band"] = LST_BAND
        lst_imgs = (
            je.ImageCollection(collection=LST_COLLECTION, ssl_verify=True)
            .filter_date(dlim=[month_start, month_end])
            .filter_resolution(ppu=ppu)
            .filter_bounds(bbox=bbox)
            .select(band=LST_BAND)
            .get_images()
        )
        lst_val = _mean_from_timeseries(lst_imgs)
        lst_c = float(lst_val - 273.15) if lst_val > 120 else float(lst_val)
    except Exception as e:
        src = "mixed"
        debug["lst_error"] = str(e)

    try:
        pr_imgs = (
            je.ImageCollection(collection=PRECIP_COLLECTION, ssl_verify=True)
            .filter_date(dlim=[week_start, week_end])
            .filter_resolution(ppu=max(10, min(ppu, 40)))
            .filter_bounds(bbox=bbox)
            .select(band=PRECIP_BAND)
            .get_images()
        )
        precip_week = _mean_from_timeseries(pr_imgs)
        debug["precip_collection"] = PRECIP_COLLECTION
        debug["precip_band"] = PRECIP_BAND
        precip_month = float(precip_week * 4.0)
        debug["precip_month_estimate"] = "week_mean * 4"
    except Exception as e:
        src = "mixed"
        debug["precip_error"] = str(e)

    if src != "jaxa":
        debug["note"] = "一部取得失敗したため fallback / 推定が混ざっています"

    return SatValues(
        ndvi=float(ndvi),
        lst=float(lst_c),
        precip_month=float(precip_month),
        precip_week_for_image=float(precip_week),
        used_month_range=(month_start, month_end),
        used_week_range=(week_start, week_end),
        source=src,
        debug=debug,
    )


# =========================
# Local SD image generation
# =========================
@st.cache_resource
def load_sd_pipeline() -> StableDiffusionPipeline:
    model = "runwayml/stable-diffusion-v1-5"
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    return pipe


def build_image_prompt(
    lat: float,
    lon: float,
    ndvi: float,
    lst_c: float,
    precip_week: float,
    pref: str,
    city: str,
    area_kind: str,
) -> str:
    ndvi_n = normalize(ndvi, 0.0, 1.0)
    lst_n = normalize(lst_c, -10.0, 45.0)
    pr_n = normalize(precip_week, 0.0, 80.0)

    if ndvi_n > 0.65:
        nature = "lush Japanese greenery, cedar forest, bamboo grove, rich vegetation"
    elif ndvi_n > 0.40:
        nature = "mixed greenery, grassland, satoyama landscape, scattered trees"
    else:
        nature = "dry grass, open land, but still Japanese terrain, minimal rocks"

    if pr_n > 0.60:
        weather = "rainy scene, visible rainfall, wet streets, puddles, soft reflections"
    elif pr_n > 0.35:
        weather = "cloudy after rain, moist air, wet ground reflections"
    else:
        weather = "clear weather, dry ground, crisp air"

    if lst_n > 0.60:
        light = "bright warm daylight, summer feeling, vivid colors"
    elif lst_n > 0.35:
        light = "mild daylight, soft sun, comfortable atmosphere"
    else:
        light = "cool daylight, overcast but bright, soft ambient light"

    if area_kind == "urban":
        place_hint = (
            "Japanese cityscape, modern buildings, narrow streets, "
            "Japanese signage, vending machines, crosswalks, "
            "clean sidewalks, subtle neon, realistic Tokyo-like vibe"
        )
    elif area_kind == "suburban":
        place_hint = (
            "Japanese suburban neighborhood, low-rise houses, small shops, "
            "utility poles and wires, quiet streets, small parks, realistic"
        )
    else:
        place_hint = (
            "Japanese countryside, rice fields, satoyama hills, "
            "small village houses, shrine torii in the distance, "
            "natural scenery, realistic"
        )

    if pref or city:
        loc = f"in {city} {pref}, Japan"
    else:
        loc = "in Japan"

    no_rocks_hint = "avoid barren rocky landscape, avoid desert, avoid canyon"

    return (
        f"photorealistic cinematic landscape {loc}, {place_hint}, "
        f"{nature}, {weather}, {light}, "
        f"ultra detailed, high realism, sharp focus, 35mm photo, {no_rocks_hint}"
    )


def generate_sd_image(
    lat: float,
    lon: float,
    ndvi: float,
    lst_c: float,
    precip_week: float,
    seed: int,
    pref: str,
    city: str,
    area_kind: str,
) -> Image.Image:
    pipe = load_sd_pipeline()
    prompt = build_image_prompt(lat, lon, ndvi, lst_c, precip_week, pref, city, area_kind)

    negative = (
        "desert, canyon, barren rocks, dry rocky wasteland, "
        "low quality, blurry, noisy, low contrast, underexposed, "
        "bad anatomy, watermark, text, logo"
    )

    dev = pipe.device.type
    generator = torch.Generator(dev).manual_seed(int(seed))

    return pipe(
        prompt,
        negative_prompt=negative,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=generator,
        height=512,
        width=512,
    ).images[0]


# =========================
# MusicGen
# =========================
@st.cache_resource
def load_musicgen(model_id: str = "facebook/musicgen-medium"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)

    model = MusicgenForConditionalGeneration.from_pretrained(
        model_id,
        use_safetensors=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()
    return processor, model, device


def _music_prompt(style: str, ndvi: float, lst_c: float, precip_month: float) -> str:
    ndvi_n = normalize(ndvi, 0.0, 1.0)
    lst_n = normalize(lst_c, -10.0, 45.0)
    pr_n = normalize(precip_month, 0.0, 500.0)

    tempo = (
        "very fast tempo" if lst_n > 0.75 else
        "fast tempo" if lst_n > 0.60 else
        "moderate tempo" if lst_n > 0.35 else
        "slow tempo"
    )

    if ndvi_n > 0.65:
        acoustic = "acoustic instruments prominent, acoustic guitar, piano, woodwinds"
    elif ndvi_n > 0.40:
        acoustic = "balanced instruments, some acoustic guitar and piano"
    else:
        acoustic = "less acoustic, more electronic elements"

    if pr_n > 0.60:
        synth = "heavy synthesizers, pads, ambient synth textures, wet reverb"
    elif pr_n > 0.35:
        synth = "some synthesizer pads, gentle ambient textures"
    else:
        synth = "minimal synth, dry and crisp sound"

    if style == "クラシック系":
        base = "instrumental cinematic classical, orchestral, strings, piano, no vocals"
    else:
        base = "instrumental rock, electric guitar riffs, bass and drums, no vocals"

    return f"{base}, {tempo}, {acoustic}, {synth}"


def _set_seed_everywhere(seed: int) -> None:
    seed = int(seed) & 0x7FFFFFFF
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fade_out(audio: np.ndarray, sr: int, fade_sec: float = 0.12) -> np.ndarray:
    n = audio.shape[0]
    fade = int(sr * fade_sec)
    if fade <= 1 or fade >= n:
        return audio
    w = np.linspace(1.0, 0.0, fade, dtype=np.float32)
    audio[-fade:, :] = audio[-fade:, :] * w[:, None]
    return audio


def synthesize_music_musicgen(
    ndvi: float,
    lst_c: float,
    precip_month: float,
    style: str,
    duration_sec: int = 12,
    model_id: str = "facebook/musicgen-medium",
    seed: int = 0,
) -> bytes:
    processor, model, device = load_musicgen(model_id)

    if seed:
        _set_seed_everywhere(seed)

    prompt = _music_prompt(style, ndvi, lst_c, precip_month)

    frame_rate = getattr(getattr(model, "config", None), "audio_encoder", None)
    frame_rate = getattr(frame_rate, "frame_rate", 50)
    max_new_tokens = int(max(1, duration_sec * float(frame_rate)))

    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        audio = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=3.0,
            max_new_tokens=max_new_tokens,
        )

    audio = audio[0].detach().float().cpu().numpy()
    if audio.ndim == 1:
        audio = audio[None, :]
    audio = np.transpose(audio, (1, 0))  # (samples, channels)

    sr = getattr(getattr(model, "config", None), "audio_encoder", None)
    sr = int(getattr(sr, "sampling_rate", 32000))

    m = float(np.max(np.abs(audio)) + 1e-8)
    audio = (audio / m).astype(np.float32)

    target = int(sr * int(duration_sec))
    if audio.shape[0] >= target:
        audio = audio[:target, :]
    else:
        pad = target - audio.shape[0]
        audio = np.pad(audio, ((0, pad), (0, 0)), mode="constant")

    audio = _fade_out(audio, sr, fade_sec=0.12)

    audio_int16 = (audio * 32767.0).clip(-32767, 32767).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, sr, audio_int16)
    return buf.getvalue()


# =========================
# Map UI
# =========================
def render_clickable_map(lat: float, lon: float, zoom: int = 12) -> Optional[Tuple[float, float]]:
    m = folium.Map(location=[lat, lon], zoom_start=zoom, control_scale=True)
    folium.Marker([lat, lon], tooltip="現在の座標").add_to(m)

    out = st_folium(m, width="100%", height=360, returned_objects=["last_clicked"])
    last = out.get("last_clicked")
    if last and "lat" in last and "lng" in last:
        return float(last["lat"]), float(last["lng"])
    return None


# =========================
# Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="JAXA Earth Soundscape (Local AI)", layout="centered")

    # ===== 見やすい配色（完成版）=====
    st.markdown(
        """
        <style>
        /* ===== アプリ背景 ===== */
        .stApp {
            background: radial-gradient(circle at 10% 10%, #fff7ae 0%, #ffffff 45%, #fff0a6 100%);
        }

        /* ===== 全体の基本文字色（明るい領域は黒） ===== */
        html, body, label, p, span, div {
            color: #111 !important;
        }

        /* ===== 入力欄は常に白背景＋黒文字 ===== */
        input, textarea {
            background-color: #ffffff !important;
            color: #111 !important;
        }

        /* ===== “現在地推定” 表示：黒背景＋白文字 ===== */
        .location-box {
            background-color: #0f172a !important;
            color: #ffffff !important;
            padding: 14px;
            border-radius: 14px;
            margin-top: 10px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        }
        .location-box strong, .location-box div {
            color: #ffffff !important;
        }

        /* ===== st.json / st.code / エラーなど “コンソールっぽい” 表示を読みやすく ===== */
        div[data-testid="stCodeBlock"],
        div[data-testid="stJson"] {
            border: 1px solid rgba(0,0,0,0.12) !important;
            border-radius: 12px !important;
            background: #fbfbfb !important;
            box-shadow: 0 6px 16px rgba(0,0,0,0.06) !important;
        }
        div[data-testid="stCodeBlock"] pre,
        div[data-testid="stJson"] pre {
            background: #fbfbfb !important;
            color: #111 !important;
        }
        div[data-testid="stJson"] * {
            color: #111 !important;
        }

        div[data-testid="stException"] {
            border-radius: 12px !important;
            border: 1px solid rgba(255,0,0,0.18) !important;
            background: #fff5f5 !important;
        }
        div[data-testid="stException"] * {
            color: #111 !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px !important;
        }
        div[data-testid="stAlert"] * {
            color: #111 !important;
        }

        /* ===== 音楽スタイル（セグメント）見やすく ===== */
        button[role="radio"] {
            background: #ffffff !important;
            color: #111 !important;
            border-radius: 12px !important;
            border: 1px solid rgba(0,0,0,0.20) !important;
            font-weight: 800 !important;
        }
        button[role="radio"][aria-checked="true"] {
            background: linear-gradient(135deg, #4caf50, #2e7d32) !important;
            color: #ffffff !important;
            border: none !important;
        }

        /* ===== 通常ボタン（黄色系） ===== */
        .stButton>button {
            background: #ffdd57 !important;
            color: #111 !important;
            border: 1px solid rgba(0,0,0,0.14) !important;
            border-radius: 12px !important;
            font-weight: 800 !important;
            padding: 0.55rem 0.9rem !important;
            box-shadow: 0 10px 18px rgba(0,0,0,0.10) !important;
        }
        .stButton>button:hover {
            background: #ffd633 !important;
            border-color: rgba(0,0,0,0.20) !important;
            transform: translateY(-1px);
        }
        .stButton>button:active {
            transform: translateY(0px);
        }

        /* ===== 生成開始ボタン（Primary）：視認性MAX ===== */
        .stButton>button[kind="primary"] {
            background: linear-gradient(135deg, #ff9800, #ff5722) !important;
            color: #ffffff !important;
            border: none !important;
            font-size: 1rem !important;
            font-weight: 900 !important;
            border-radius: 14px !important;
            padding: 0.75rem 1rem !important;
            box-shadow: 0 10px 20px rgba(255,87,34,0.35) !important;
        }
        .stButton>button[kind="primary"]:hover {
            background: linear-gradient(135deg, #ffa726, #ff7043) !important;
        }

        /* ===== disabled ===== */
        .stButton>button:disabled {
            background: #e5e7eb !important;
            color: #6b7280 !important;
            border: 1px solid rgba(0,0,0,0.10) !important;
            box-shadow: none !important;
            cursor: not-allowed !important;
        }

        /* ===== “スマホ枠” ===== */
        .phone {
            max-width: 520px;
            margin: 0 auto;
            border-radius: 36px;
            padding: 18px 16px;
            border: 3px solid #ffe261;
            box-shadow: 0 14px 30px rgba(0,0,0,0.18);
            background: rgba(255, 255, 255, 0.96);
        }
        .title {
            font-size: 1.6rem;
            font-weight: 900;
            color: #2b2200 !important;
            margin-bottom: 0.2rem;
        }
        .sub {
            color: #4a3a00 !important;
            font-size: 0.95rem;
            margin-bottom: 0.9rem;
        }
        </style>

        <div class="phone">
          <div class="title">🌍✨ JAXA Earth Soundscape</div>
          <div class="sub">地図クリック → 座標 → 衛星データ → 日本っぽい画像＋MusicGen音楽（ローカル）</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "lat" not in st.session_state:
        st.session_state.lat = 35.680959
    if "lon" not in st.session_state:
        st.session_state.lon = 139.767306
    if "geo" not in st.session_state:
        st.session_state.geo = None
    if "pref" not in st.session_state:
        st.session_state.pref = ""
    if "city" not in st.session_state:
        st.session_state.city = ""
    if "area_kind" not in st.session_state:
        st.session_state.area_kind = "urban"

    with st.container(border=True):
        st.subheader("📍 地図クリックで座標取得")

        clicked = render_clickable_map(st.session_state.lat, st.session_state.lon, zoom=12)
        if clicked:
            st.session_state.lat, st.session_state.lon = clicked

        st.caption("※ 右側の入力欄から直接入力もできます（小数第6位まで対応）")

        c1, c2 = st.columns(2)
        with c1:
            st.session_state.lat = st.number_input(
                "緯度 (Latitude)",
                min_value=-90.0,
                max_value=90.0,
                value=float(st.session_state.lat),
                step=0.000001,
                format="%.6f",
            )
        with c2:
            st.session_state.lon = st.number_input(
                "経度 (Longitude)",
                min_value=-180.0,
                max_value=180.0,
                value=float(st.session_state.lon),
                step=0.000001,
                format="%.6f",
            )

        month = st.date_input("作成する月（この1か月前のNDVI/LST、1週間前の降水）", value=date.today().replace(day=1))

        if st.button("🗾 県名・都市名を取得（自動）", use_container_width=True):
            with st.spinner("県名・都市名を取得中..."):
                try:
                    geo = reverse_geocode_nominatim(st.session_state.lat, st.session_state.lon)
                    pref, city, display = parse_pref_city(geo)
                    st.session_state.geo = geo
                    st.session_state.pref = pref
                    st.session_state.city = city
                    st.session_state.area_kind = classify_area(st.session_state.lat, st.session_state.lon, geo)
                    st.success("取得できました！")
                    st.write({"pref": pref, "city": city, "area_kind": st.session_state.area_kind, "display_name": display})
                except Exception as e:
                    st.session_state.geo = None
                    st.session_state.pref = ""
                    st.session_state.city = ""
                    st.session_state.area_kind = classify_area(st.session_state.lat, st.session_state.lon, None)
                    st.warning(f"取得に失敗しました（ネットワーク/制限の可能性）。画像は座標のみで生成します。詳細: {e}")

        # ★ここが「現在地推定」黒背景＋白文字（完成版）
        current_loc = f"{st.session_state.city} {st.session_state.pref}".strip() or "(未取得)"
        st.markdown(
            f"""
            <div class="location-box">
                <div><strong>現在地推定：</strong>{current_loc}</div>
                <div><strong>都市判定：</strong>{st.session_state.area_kind}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        style = st.segmented_control("音楽スタイル", options=["クラシック系", "ロック系"], default="クラシック系")
        duration_sec = st.slider("音楽の長さ（秒）", min_value=6, max_value=20, value=12, step=1)

        seed = st.number_input("生成シード（再現性/変化）", min_value=0, max_value=2_000_000_000, value=0, step=1)
        if seed == 0:
            seed = int((abs(st.session_state.lat) * 1000000 + abs(st.session_state.lon) * 1000000 + month.year * 100 + month.month) % 2_000_000_000)

        generate = st.button("画像と音楽を作成", type="primary", use_container_width=True)

    if generate and style:
        with st.spinner("衛星データ取得 → 作品生成中...（初回はモデルDLで時間がかかります）"):
            sat = fetch_satellite_values(st.session_state.lat, st.session_state.lon, month)

            image = generate_sd_image(
                st.session_state.lat,
                st.session_state.lon,
                sat.ndvi,
                sat.lst,
                sat.precip_week_for_image,
                seed=seed,
                pref=st.session_state.pref,
                city=st.session_state.city,
                area_kind=st.session_state.area_kind,
            )

            wav = synthesize_music_musicgen(
                sat.ndvi,
                sat.lst,
                sat.precip_month,
                style=style,
                duration_sec=int(duration_sec),
                model_id="facebook/musicgen-medium",
                seed=int(seed),
            )

        st.success("生成が完了しました！")

        st.write(
            {
                "source": sat.source,
                "target_month": month.strftime("%Y-%m"),
                "used_month_range_for_ndvi_lst": sat.used_month_range,
                "used_week_range_for_precip": sat.used_week_range,
                "music_style": style,
                "duration_sec": int(duration_sec),
                "seed": int(seed),
                "ndvi_monthly": round(sat.ndvi, 4),
                "lst_monthly_celsius": round(sat.lst, 2),
                "precip_month_estimated": round(sat.precip_month, 2),
                "precip_week_mean": round(sat.precip_week_for_image, 2),
                "pref_city": f"{st.session_state.city} {st.session_state.pref}".strip(),
                "area_kind": st.session_state.area_kind,
            }
        )

        with st.expander("デバッグ情報（衛星データ取得）"):
            st.json(sat.debug)

        if st.session_state.geo:
            with st.expander("デバッグ情報（逆ジオコーディング）"):
                st.json(st.session_state.geo)

        st.image(image, caption="衛星データ＋地域特徴（都市/田舎/県市）を反映した日本っぽいランドスケープ", use_container_width=True)
        st.audio(wav, format="audio/wav")


if __name__ == "__main__":
    main()
