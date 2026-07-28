import base64
import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# Tema

BG_DARK = "#0B0E14"
BG_CARD = "#161A26"
BG_CARD_BORDER = "#262B3D"
ACCENT_GREEN = "#A4CE39"
ACCENT_GREEN_DARK = "#7A9C2A"
ACCENT_BLUE = "#4FA8E0"
ACCENT_PURPLE = "#A374DB"
TEXT_LIGHT = "#E8E9ED"
TEXT_MUTED = "#9AA0AC"
RED_PAHALI = "#E05252"

UI_SCALE = 0.85

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {BG_DARK};
        color: {TEXT_LIGHT};
    }}
    @media (min-width: 900px) {{
        .stApp {{
            zoom: {UI_SCALE};
        }}
        button[data-testid="collapsedControl"] {{
            display: none;
        }}
    }}
    .scout-mobile-menu-hint {{
        display: none;
    }}
    @media (max-width: 899px) {{
        .scout-mobile-menu-hint {{
            display: flex !important;
            align-items: center;
            gap: 9px;
            background-color: {BG_CARD};
            border: 1px solid {ACCENT_GREEN};
            border-left: 4px solid {ACCENT_GREEN};
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 14px;
            color: {TEXT_LIGHT};
            font-size: 12.5px;
            line-height: 1.45;
        }}
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"],
        button[data-testid="collapsedControl"],
        button[data-testid="stSidebarCollapseButton"] {{
            border: 1px solid {ACCENT_GREEN} !important;
            border-radius: 6px !important;
            background-color: {BG_CARD} !important;
        }}
        [data-testid="stSidebarCollapsedControl"] svg,
        [data-testid="collapsedControl"] svg,
        button[data-testid="stSidebarCollapseButton"] svg {{
            fill: {ACCENT_GREEN} !important;
            color: {ACCENT_GREEN} !important;
        }}
        .scout-header-wrap {{
            flex-direction: column !important;
            align-items: stretch !important;
            overflow: visible !important;
            padding: 14px 16px !important;
        }}
        .scout-header-wrap > div {{
            flex: none !important;
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            margin: 0 0 12px 0 !important;
        }}
        .scout-header-left {{
            gap: 12px !important;
        }}
        .scout-title {{
            font-size: 30px !important;
        }}
        .scout-subtitle {{
            font-size: 11px !important;
            letter-spacing: 0.3px !important;
            line-height: 1.35 !important;
        }}
        .scout-chip-row {{
            flex-wrap: wrap !important;
            gap: 14px !important;
            justify-content: flex-start !important;
        }}
        .scout-chip {{
            min-width: 66px !important;
        }}
        .scout-chip-label {{
            font-size: 9px !important;
        }}
        .scout-badge-row {{
            font-size: 11px !important;
        }}
    }}
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    .stRadio label, .stCheckbox label, .stSlider label {{
        color: {TEXT_LIGHT} !important;
    }}
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] {{
        color: {TEXT_MUTED} !important;
    }}
    [data-baseweb="select"] > div,
    [data-baseweb="popover"] li {{
        background-color: {BG_CARD} !important;
        color: {TEXT_LIGHT} !important;
    }}
    [data-testid="stExpander"] details {{
        background-color: {BG_CARD};
        border-color: {BG_CARD_BORDER};
    }}
    [data-testid="stExpander"] summary p {{
        color: {TEXT_LIGHT} !important;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {BG_CARD};
    }}
    div[data-testid="stMetric"] {{
        background-color: {BG_CARD};
        border: 1px solid {BG_CARD_BORDER};
        border-radius: 8px;
        padding: 14px 10px;
    }}
    div[data-testid="stMetric"] label {{
        color: {TEXT_MUTED} !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {ACCENT_GREEN} !important;
        font-weight: 700;
    }}
    .scout-card {{
        background-color: {BG_CARD};
        border: 1px solid {BG_CARD_BORDER};
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
    }}
    .scout-card-title {{
        color: {TEXT_LIGHT};
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
        text-transform: uppercase;
        border-bottom: 1px solid {BG_CARD_BORDER};
        padding-bottom: 8px;
    }}
    .scout-section-title {{
        color: {TEXT_LIGHT};
        font-weight: 700;
        font-size: 14.5px;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        border-bottom: 1px solid {BG_CARD_BORDER};
        padding-bottom: 7px;
        margin: 6px 0 10px 0;
    }}
    .scout-section-end {{
        height: 18px;
    }}
    .scout-header-wrap {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(90deg, {BG_CARD} 0%, {BG_DARK} 100%);
        border: 1px solid {BG_CARD_BORDER};
        border-bottom: 2px solid {ACCENT_GREEN};
        border-radius: 12px;
        padding: 10px 26px;
        margin-bottom: 22px;
        overflow: hidden;
    }}
    .scout-header-left {{
        display: flex;
        align-items: center;
        gap: 18px;
    }}
    .scout-title {{
        color: {ACCENT_GREEN};
        font-weight: 900;
        font-size: 46px;
        letter-spacing: 1px;
        line-height: 1.0;
        text-shadow: 0 0 18px rgba(164, 206, 57, 0.45);
        margin: 0;
    }}
    .scout-subtitle {{
        color: {TEXT_LIGHT};
        font-size: 17px;
        font-weight: 600;
        letter-spacing: 1.5px;
        margin-top: 2px;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {BG_CARD};
        border-radius: 6px;
        color: {TEXT_MUTED};
        padding: 8px 18px;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {ACCENT_GREEN} !important;
        color: #0B0E14 !important;
        font-weight: 700;
    }}
    .scout-title-white {{
        color: {TEXT_LIGHT};
    }}
    .scout-chip-row {{
        display: flex;
        gap: 22px;
        margin-top: 14px;
        flex-wrap: wrap;
    }}
    .scout-chip {{
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
        min-width: 78px;
    }}
    .scout-chip-label {{
        color: {TEXT_LIGHT};
        font-size: 11px;
        font-weight: 700;
        text-align: center;
        letter-spacing: 0.3px;
        line-height: 1.25;
    }}
    .scout-badge-list {{
        display: flex;
        flex-direction: column;
        gap: 9px;
    }}
    .scout-badge-row {{
        display: flex;
        align-items: center;
        gap: 9px;
        color: {TEXT_LIGHT};
        font-size: 12px;
        font-weight: 600;
    }}
    .scout-version-pill {{
        display: inline-block;
        margin-top: 10px;
        border: 1px solid {ACCENT_GREEN};
        color: {ACCENT_GREEN};
        border-radius: 6px;
        padding: 5px 14px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}
    .scout-stat-row {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 20px;
    }}
    .scout-stat-card {{
        background-color: {BG_CARD};
        border: 1px solid {BG_CARD_BORDER};
        border-radius: 8px;
        padding: 14px 16px;
        display: flex;
        align-items: center;
        gap: 10px;
        flex: 1;
        min-width: 150px;
    }}
    .scout-stat-label {{
        color: {TEXT_MUTED};
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        white-space: nowrap;
    }}
    .scout-stat-value {{
        font-size: 21px;
        font-weight: 800;
        white-space: nowrap;
    }}
    .scout-welcome-card {{
        background-color: {BG_CARD};
        border: 1px solid {BG_CARD_BORDER};
        border-radius: 10px;
        padding: 22px;
    }}
    .scout-feature-card {{
        background-color: {BG_CARD};
        border: 1px solid {BG_CARD_BORDER};
        border-left: 3px solid {ACCENT_GREEN};
        border-radius: 8px;
        padding: 16px;
        height: 100%;
    }}
    .scout-feature-card-title {{
        color: {TEXT_LIGHT};
        font-weight: 700;
        font-size: 14px;
        margin: 8px 0 6px 0;
    }}
    .scout-feature-card-desc {{
        color: {TEXT_MUTED};
        font-size: 12.5px;
        line-height: 1.5;
    }}
    .scout-update-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid {BG_CARD_BORDER};
        font-size: 13px;
    }}
    .scout-update-row:last-child {{
        border-bottom: none;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        font-size: 14px;
        font-weight: 600;
        padding: 6px 4px;
    }}
</style>
"""

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_CARD,
    plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_LIGHT, size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    bargap=0.35,
    bargroupgap=0.15,
    dragmode=False,
)

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": False,
    "doubleClick": False,
    "showTips": False,
    "modeBarButtonsToRemove": [
        "zoom2d", "pan2d", "select2d", "lasso2d",
        "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
        "zoomInGeo", "zoomOutGeo", "resetGeo",
        "hoverClosestGeo", "hoverClosestCartesian", "hoverCompareCartesian",
    ],
}


def render_chart(fig, **kwargs):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(dragmode=False)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, **kwargs)


def scout_card_start(title: str):
    st.markdown(f'<div class="scout-section-title">{title}</div>', unsafe_allow_html=True)


def scout_card_end():
    st.markdown('<div class="scout-section-end"></div>', unsafe_allow_html=True)


def format_euro(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M €"
    elif value >= 1_000:
        return f"{value / 1_000:.0f}K €"
    else:
        return f"{value:,.0f} €"


def format_euro_tam(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:,.0f}".replace(",", ".") + " €"


# Konfigürasyon

class Config:
    DATA_DIR = Path(".")

    ML_READY_FILE = "02_ML_Ready_v2.csv"
    PLAYER_REFERENCE_FILE = "02_Player_Reference.csv"
    BEST_MODEL_FILE = "best_model.pkl"
    X_TRAIN_FILE = "X_train.csv"
    X_TEST_FILE = "X_test.csv"
    REFERENCE_TEST_FILE = "reference_test.csv"

    TARGET_COLUMN = "PiyasaDegeri"
    NUM_PRICE_SEGMENTS = 8

    TEAM_TACTIC_FEATURES = [
        "buildUpPlaySpeed", "buildUpPlayDribbling", "buildUpPlayPassing",
        "chanceCreationPassing", "chanceCreationCrossing", "chanceCreationShooting",
        "defencePressure", "defenceAggression", "defenceTeamWidth",
    ]

    SIMILARITY_FEATURES = [
        "crossing", "finishing", "heading_accuracy", "short_passing",
        "volleys", "dribbling", "curve", "free_kick_accuracy",
        "long_passing", "ball_control", "acceleration", "sprint_speed",
        "agility", "reactions", "balance", "shot_power", "jumping",
        "stamina", "strength", "long_shots", "aggression", "interceptions",
        "positioning", "vision", "penalties", "marking", "standing_tackle",
        "sliding_tackle", "gk_diving", "gk_handling", "gk_kicking",
        "gk_positioning", "gk_reflexes",
    ]

    TACTIC_TO_SKILL_MAP = {
        "buildUpPlaySpeed": ["sprint_speed", "acceleration"],
        "buildUpPlayDribbling": ["dribbling", "ball_control"],
        "buildUpPlayPassing": ["short_passing", "long_passing", "vision"],
        "chanceCreationPassing": ["vision", "short_passing"],
        "chanceCreationCrossing": ["crossing"],
        "chanceCreationShooting": ["finishing", "shot_power", "long_shots"],
        "defencePressure": ["stamina", "interceptions"],
        "defenceAggression": ["aggression", "standing_tackle", "sliding_tackle"],
        "defenceTeamWidth": ["stamina", "acceleration"],
    }

    POSITION_GROUPS = {
        "KALECI": ["position_GK"],
        "STOPER": ["position_CB", "position_LCB", "position_RCB"],
        "BEK": ["position_LB", "position_RB", "position_LWB", "position_RWB"],
        "MERKEZ_ORTA_SAHA": ["position_CM", "position_LCM", "position_RCM"],
        "ON_LIBERO": ["position_CAM", "position_LAM", "position_RAM"],
        "KANAT": ["position_LM", "position_RM", "position_LW", "position_RW"],
        "FORVET": ["position_ST", "position_LST", "position_RST",
                   "position_LF", "position_RF", "position_SS"],
    }

    GROUP_DISPLAY_NAMES = {
        "KALECI": "Kaleci", "STOPER": "Stoper", "BEK": "Bek",
        "MERKEZ_ORTA_SAHA": "Merkez Orta Saha", "ON_LIBERO": "10 Numara",
        "KANAT": "Kanat", "FORVET": "Forvet",
    }

    POSITION_DISPLAY = {
        "position_GK": "Kaleci",
        "position_CB": "Stoper", "position_LCB": "Sol Stoper", "position_RCB": "Sağ Stoper",
        "position_LB": "Sol Bek", "position_RB": "Sağ Bek",
        "position_LWB": "Sol Kanat Beki", "position_RWB": "Sağ Kanat Beki",
        "position_CM": "Merkez Orta Saha", "position_LCM": "Sol Merkez Orta Saha",
        "position_RCM": "Sağ Merkez Orta Saha",
        "position_CAM": "10 Numara", "position_LAM": "10 Numara", "position_RAM": "10 Numara",
        "position_LM": "Sol Orta Saha", "position_RM": "Sağ Orta Saha",
        "position_LW": "Sol Kanat", "position_RW": "Sağ Kanat",
        "position_ST": "Santrafor", "position_LST": "Santrafor", "position_RST": "Santrafor",
        "position_LF": "Sol Forvet", "position_RF": "Sağ Forvet",
        "position_SS": "İkinci Forvet",
    }

    COUNTRY_TR = {
        "France": "Fransa", "Italy": "İtalya", "Spain": "İspanya",
        "England": "İngiltere", "Portugal": "Portekiz", "Germany": "Almanya",
        "Netherlands": "Hollanda", "Belgium": "Belçika", "Poland": "Polonya",
        "Scotland": "İskoçya", "Switzerland": "İsviçre",
    }

    BENZERLIK_AGIRLIGI = 0.70
    TAKIM_UYUM_AGIRLIGI = 0.30

    DISPLAY_NAMES = {
        "buildUpPlaySpeed": "Hızlı Oyun Kurma", "buildUpPlayDribbling": "Top Sürerek İlerleme",
        "buildUpPlayPassing": "Pas Odaklı Oyun Kurma", "chanceCreationPassing": "Pas ile Fırsat Yaratma",
        "chanceCreationCrossing": "Orta ile Fırsat Yaratma", "chanceCreationShooting": "Şut Odaklı Fırsat Yaratma",
        "defencePressure": "Üst Düzey Pres", "defenceAggression": "Savunma Agresifliği",
        "defenceTeamWidth": "Savunma Genişliği",
        "overall_rating": "Genel Seviye", "potential": "Potansiyel", "Age": "Yaş",
        "matches_played": "Maç Sayısı", "goals": "Gol", "assists": "Asist",
        "attack_score": "Hücum Puanı", "midfield_score": "Orta Saha Puanı",
        "defense_score": "Savunma Puanı", "gk_score": "Kalecilik Puanı",
        "height (cm)": "Boy (cm)", "weight (kg)": "Kilo (kg)",
        "crossing": "Orta Yapma", "finishing": "Bitiricilik", "heading_accuracy": "Kafa Vuruşu",
        "short_passing": "Kısa Pas", "volleys": "Vole", "dribbling": "Top Sürme", "curve": "Falso",
        "free_kick_accuracy": "Serbest Vuruş", "long_passing": "Uzun Pas", "ball_control": "Top Kontrolü",
        "acceleration": "İvmelenme", "sprint_speed": "Sprint Hızı", "agility": "Çeviklik",
        "reactions": "Reaksiyon", "balance": "Denge", "shot_power": "Şut Gücü", "jumping": "Zıplama",
        "stamina": "Dayanıklılık", "strength": "Fiziksel Güç", "long_shots": "Uzaktan Şut",
        "aggression": "Mücadelecilik", "interceptions": "Top Kapma", "positioning": "Konumlanma",
        "vision": "Vizyon", "penalties": "Penaltı", "marking": "Adam Tutma",
        "standing_tackle": "Ayakta Müdahale", "sliding_tackle": "Kayarak Müdahale",
        "gk_diving": "Kaleci - Plonjon", "gk_handling": "Kaleci - Top Tutma", "gk_kicking": "Kaleci - Vuruş",
        "gk_positioning": "Kaleci - Konumlanma", "gk_reflexes": "Kaleci - Refleks",
    }

    RADAR_KATEGORILERI = {
        "Şut": ["finishing", "shot_power", "long_shots", "volleys", "penalties"],
        "Fiziksel": ["strength", "stamina", "jumping", "aggression"],
        "Pas": ["short_passing", "long_passing", "vision", "curve"],
        "Top Sürme": ["dribbling", "ball_control", "agility", "balance"],
        "Hız": ["sprint_speed", "acceleration"],
        "Savunma": ["marking", "standing_tackle", "sliding_tackle", "interceptions"],
    }

    RADAR_KATEGORILERI_KALECI = {
        "Refleks": ["gk_reflexes"],
        "Plonjon": ["gk_diving"],
        "Top Tutma": ["gk_handling"],
        "Ayak Oyunu": ["gk_kicking"],
        "Konumlanma": ["gk_positioning"],
        "Fiziksel": ["strength", "jumping", "agility", "reactions"],
    }

    TEKNIK_HUCUM = ["heading_accuracy", "ball_control", "shot_power", "short_passing",
                    "crossing", "curve", "long_passing", "free_kick_accuracy",
                    "penalties", "volleys", "vision", "finishing", "dribbling", "long_shots"]
    FIZIKSEL_HAREKET = ["jumping", "stamina", "aggression", "agility", "reactions",
                        "strength", "sprint_speed", "acceleration", "balance", "positioning"]
    SAVUNMA = ["sliding_tackle", "standing_tackle", "marking"]

    KALECILIK = ["gk_reflexes", "gk_diving", "gk_handling", "gk_positioning", "gk_kicking"]
    KALECI_AYAK_OYUNU = ["short_passing", "long_passing", "ball_control", "vision", "curve"]

    SIMILARITY_FEATURES_KALECI = [
        "gk_reflexes", "gk_diving", "gk_handling", "gk_positioning", "gk_kicking",
        "reactions", "jumping", "strength", "agility", "balance", "positioning",
        "short_passing", "long_passing",
    ]

    TACTIC_TO_SKILL_MAP_KALECI = {
        "buildUpPlaySpeed": ["gk_kicking"],
        "buildUpPlayPassing": ["gk_kicking", "short_passing"],
        "buildUpPlayDribbling": ["gk_handling", "ball_control"],
        "defencePressure": ["gk_positioning", "acceleration"],
        "defenceAggression": ["gk_reflexes", "gk_positioning"],
        "defenceTeamWidth": ["gk_diving", "gk_positioning"],
    }


# Görseller

ASSETS_BASE = "https://raw.githubusercontent.com/Krauwnsm/scoutai-assets-2026/main"

COUNTRY_FLAG_FILES = {
    "Belgium": "Flag_of_Belgium.png",
    "England": "Flag_of_England.png",
    "France": "Flag_of_France.png",
    "Germany": "Flag_of_Germany.png",
    "Italy": "Flag_of_Italy.png",
    "Netherlands": "Flag_of_the_Netherlands.png",
    "Poland": "Flag_of_Poland.png",
    "Scotland": "Flag_of_Scotland.png",
    "Spain": "Flag_of_Spain.png",
    "Switzerland": "Flag_of_Switzerland.png",
}

LEAGUE_LOGO_FILES = {
    "Belgium Jupiler League": "jupiler league.png",
    "England Premier League": "premier league.png",
    "France Ligue 1": "france ligue 1.png",
    "Germany 1. Bundesliga": "bundesliga.png",
    "Italy Serie A": "seria a.png",
    "Netherlands Eredivisie": "eredivisie.png",
    "Poland Ekstraklasa": "ekstraklasa.png",
    "Portugal Liga ZON Sagres": "liga zon sagres.png",
    "Scotland Premier League": "scotland premier league.png",
    "Spain LIGA BBVA": "la liga.png",
    "Switzerland Super League": "switzerland super league.png",
}


DEFAULT_PLAYER_URL = f"{ASSETS_BASE}/players/default_player.png"
DEFAULT_TEAM_URL = f"{ASSETS_BASE}/logos/default_team.png"

AYNI_ISIMLI_FOTO_IDLERI = {
    2802, 17299, 19249, 25382, 26564, 27303, 27427, 28467, 28480, 31303,
    34305, 38424, 38484, 40196, 47559, 49451, 49940, 56686, 78908, 97363,
    101091, 101103, 103905, 105828, 114971, 115519, 119702, 128045, 150041, 162497,
    163838, 164241, 171899, 172157, 173630, 175646, 181205, 181211, 186687, 188652,
    196824, 198566, 202639, 208077, 213816, 242469, 246177, 246201, 246422, 265275,
    267863, 280350, 362665, 375790, 388456, 422808, 504033, 533212, 568911, 570432,
}


def _slugify(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.lower().replace(".", "")
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name


@st.cache_data(ttl=86400, show_spinner=False)
def _find_asset_url(folder: str, base_filename: str, extensions: tuple):
    for ext in extensions:
        url = f"{ASSETS_BASE}/{folder}/{base_filename}{ext}"
        try:
            resp = requests.get(url, timeout=3, stream=True)
            if resp.status_code == 200:
                return url
        except requests.RequestException:
            continue
    return None


def get_player_photo_url(player_name: str, player_api_id=None):
    if player_api_id is not None:
        try:
            pid = int(player_api_id)
        except (TypeError, ValueError):
            pid = None
        if pid in AYNI_ISIMLI_FOTO_IDLERI:
            url = _find_asset_url("players", str(pid), (".png", ".jpg"))
            return url or DEFAULT_PLAYER_URL

    url = _find_asset_url("players", _slugify(player_name), (".png", ".jpg"))
    return url or DEFAULT_PLAYER_URL


def get_player_photo_from_row(row):
    return get_player_photo_url(row["player_name"], row.get("player_api_id"))


def get_team_logo_url(team_name: str):
    url = _find_asset_url("logos", _slugify(team_name), (".png", ".jpg"))
    return url or DEFAULT_TEAM_URL


def get_country_flag_url(country: str):
    dosya = COUNTRY_FLAG_FILES.get(country)
    return f"{ASSETS_BASE}/country/{dosya}" if dosya else None


def get_league_logo_url(league: str):
    dosya = LEAGUE_LOGO_FILES.get(league)
    return f"{ASSETS_BASE}/league/{dosya}" if dosya else None


# SVG

def _flatten_html(html: str) -> str:
    return " ".join(line.strip() for line in html.strip().splitlines())


def pitch_network_svg() -> str:
    dugumler = [(70, 108), (150, 42), (150, 108), (260, 75), (340, 42), (340, 108), (430, 75)]
    baglantilar = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)]

    cizgiler = "".join(
        f'<line x1="{dugumler[a][0]}" y1="{dugumler[a][1]}" x2="{dugumler[b][0]}" y2="{dugumler[b][1]}" '
        f'stroke="{ACCENT_GREEN}" stroke-width="1.3" opacity="0.55"/>'
        for a, b in baglantilar
    )
    noktalar = "".join(
        f'<circle cx="{x}" cy="{y}" r="5" fill="{ACCENT_GREEN}" opacity="0.9"/>'
        for x, y in dugumler
    )

    return f"""
    <svg width="100%" height="120" viewBox="0 0 500 150" preserveAspectRatio="xMidYMid meet"
         xmlns="http://www.w3.org/2000/svg">
        <rect x="20" y="15" width="460" height="120" rx="6" fill="none" stroke="{ACCENT_GREEN}" stroke-width="1.4" opacity="0.32"/>
        <line x1="250" y1="15" x2="250" y2="135" stroke="{ACCENT_GREEN}" stroke-width="1.2" opacity="0.32"/>
        <circle cx="250" cy="75" r="30" fill="none" stroke="{ACCENT_GREEN}" stroke-width="1.2" opacity="0.32"/>
        {cizgiler}
        {noktalar}
    </svg>
    """


def hexagon_ball_logo_svg(size: int = 78) -> str:
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <polygon points="50,4 90,27 90,73 50,96 10,73 10,27"
                 fill="{BG_CARD}" stroke="{ACCENT_GREEN}" stroke-width="3.5"/>
        <circle cx="50" cy="50" r="28" fill="none" stroke="{TEXT_LIGHT}" stroke-width="2.5"/>
        <polygon points="50,32 61,40 57,53 43,53 39,40" fill="{TEXT_LIGHT}"/>
        <path d="M50,32 L50,22.5 M61,40 L70,33.5 M57,53 L61,63 M43,53 L39,63 M39,40 L30,33.5"
              stroke="{TEXT_LIGHT}" stroke-width="2" fill="none"/>
    </svg>
    """


def _icon_brain(color: str) -> str:
    return (f'<circle cx="20" cy="20" r="16" fill="none" stroke="{color}" stroke-width="2"/>'
            f'<path d="M20 8 L20 32 M12 14 Q20 18 12 26 M28 14 Q20 18 28 26" '
            f'stroke="{color}" stroke-width="1.6" fill="none"/>')


def _icon_chart_up(color: str) -> str:
    return (f'<polyline points="6,30 15,20 22,25 34,8" fill="none" stroke="{color}" stroke-width="2.4" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
            f'<polyline points="25,8 34,8 34,17" fill="none" stroke="{color}" stroke-width="2.4" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')


def _icon_target(color: str) -> str:
    return (f'<circle cx="20" cy="20" r="16" fill="none" stroke="{color}" stroke-width="2"/>'
            f'<circle cx="20" cy="20" r="9" fill="none" stroke="{color}" stroke-width="2"/>'
            f'<circle cx="20" cy="20" r="2.5" fill="{color}"/>')


def _icon_network(color: str) -> str:
    return (f'<circle cx="10" cy="10" r="4" fill="{color}"/><circle cx="30" cy="10" r="4" fill="{color}"/>'
            f'<circle cx="20" cy="30" r="4" fill="{color}"/>'
            f'<line x1="10" y1="10" x2="30" y2="10" stroke="{color}" stroke-width="1.6"/>'
            f'<line x1="10" y1="10" x2="20" y2="30" stroke="{color}" stroke-width="1.6"/>'
            f'<line x1="30" y1="10" x2="20" y2="30" stroke="{color}" stroke-width="1.6"/>')


def _icon_pie(color: str) -> str:
    return (f'<circle cx="20" cy="20" r="16" fill="none" stroke="{color}" stroke-width="2"/>'
            f'<path d="M20 20 L20 4 A16 16 0 0 1 34 28 Z" fill="{color}" opacity="0.85"/>')


def _icon_people(color: str) -> str:
    return (f'<circle cx="14" cy="13" r="6" fill="none" stroke="{color}" stroke-width="2"/>'
            f'<path d="M4 34 Q4 22 14 22 Q24 22 24 34" fill="none" stroke="{color}" stroke-width="2"/>'
            f'<circle cx="27" cy="15" r="5" fill="none" stroke="{color}" stroke-width="1.8" opacity="0.7"/>'
            f'<path d="M20 34 Q20 25 27 25 Q34 25 34 34" fill="none" stroke="{color}" stroke-width="1.8" opacity="0.7"/>')


def _icon_shield(color: str) -> str:
    return (f'<path d="M20 4 L34 10 L34 20 Q34 30 20 36 Q6 30 6 20 L6 10 Z" '
            f'fill="none" stroke="{color}" stroke-width="2.2"/>'
            f'<path d="M13 19 L18 25 L28 13" fill="none" stroke="{color}" stroke-width="2.2" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')


def _icon_globe(color: str) -> str:
    return (f'<circle cx="20" cy="20" r="16" fill="none" stroke="{color}" stroke-width="2"/>'
            f'<ellipse cx="20" cy="20" rx="7" ry="16" fill="none" stroke="{color}" stroke-width="1.6"/>'
            f'<line x1="4" y1="20" x2="36" y2="20" stroke="{color}" stroke-width="1.6"/>'
            f'<path d="M7 12 Q20 18 33 12" fill="none" stroke="{color}" stroke-width="1.2" opacity="0.7"/>'
            f'<path d="M7 28 Q20 22 33 28" fill="none" stroke="{color}" stroke-width="1.2" opacity="0.7"/>')


def _icon_cpu(color: str) -> str:
    return (f'<rect x="11" y="11" width="18" height="18" rx="2" fill="none" stroke="{color}" stroke-width="2"/>'
            f'<rect x="16" y="16" width="8" height="8" fill="{color}" opacity="0.8"/>'
            f'<line x1="20" y1="2" x2="20" y2="11" stroke="{color}" stroke-width="2"/>'
            f'<line x1="20" y1="29" x2="20" y2="38" stroke="{color}" stroke-width="2"/>'
            f'<line x1="2" y1="20" x2="11" y2="20" stroke="{color}" stroke-width="2"/>'
            f'<line x1="29" y1="20" x2="38" y2="20" stroke="{color}" stroke-width="2"/>')


def _icon_database(color: str) -> str:
    return (f'<ellipse cx="20" cy="9" rx="14" ry="5" fill="none" stroke="{color}" stroke-width="2"/>'
            f'<path d="M6 9 L6 31 Q6 36 20 36 Q34 36 34 31 L34 9" fill="none" stroke="{color}" stroke-width="2"/>'
            f'<path d="M6 20 Q6 25 20 25 Q34 25 34 20" fill="none" stroke="{color}" stroke-width="1.6" opacity="0.7"/>')


def _icon_ruler(color: str) -> str:
    return (f'<rect x="4" y="14" width="32" height="12" rx="2" fill="none" stroke="{color}" stroke-width="2"/>'
            f'<line x1="11" y1="14" x2="11" y2="20" stroke="{color}" stroke-width="1.6"/>'
            f'<line x1="18" y1="14" x2="18" y2="22" stroke="{color}" stroke-width="1.6"/>'
            f'<line x1="25" y1="14" x2="25" y2="20" stroke="{color}" stroke-width="1.6"/>'
            f'<line x1="32" y1="14" x2="32" y2="22" stroke="{color}" stroke-width="1.6"/>')


FEATURE_CHIPS = [
    ("Makine Öğrenmesi", _icon_brain),
    ("Değer Tahmini", _icon_chart_up),
    ("Akıllı Scout", _icon_target),
    ("Kosinüs Benzerliği", _icon_network),
    ("Performans Analitiği", _icon_pie),
]


def feature_chips_html() -> str:
    chips = ""
    for label, icon_fn in FEATURE_CHIPS:
        icon = icon_fn(ACCENT_GREEN)
        chips += (
            f'<div class="scout-chip">'
            f'<svg width="34" height="34" viewBox="0 0 40 40">{icon}</svg>'
            f'<div class="scout-chip-label">{label}</div>'
            f'</div>'
        )
    return f'<div class="scout-chip-row">{chips}</div>'


def info_badges_html(model_adi: str) -> str:
    satirlar = [
        (_icon_database, "EUROPEAN SOCCER DATABASE", ACCENT_BLUE),
        (_icon_shield, "TRANSFERMARKT", TEXT_LIGHT),
        (_icon_cpu, model_adi.upper(), ACCENT_PURPLE),
        (_icon_network, "KOSİNÜS BENZERLİĞİ", ACCENT_GREEN),
    ]
    rows = "".join(
        f'<div class="scout-badge-row">'
        f'<svg width="16" height="16" viewBox="0 0 40 40">{icon_fn(renk)}</svg>'
        f'<b>{ust}</b></div>'
        for icon_fn, ust, renk in satirlar
    )
    return (
        f'<div class="scout-badge-list">{rows}</div>'
        f'<div class="scout-version-pill">SÜRÜM 1.0</div>'
    )


BILINEN_PIPELINE_GORSELLERI = {
    "06_prediction_vs_actual.png", "06_residual_plot.png",
    "06_residual_distribution.png", "06_feature_importance.png",
}


def find_hero_image_file(config: Config):
    for desen in ("*.png", "*.jpg", "*.jpeg"):
        for path in sorted(config.DATA_DIR.glob(desen)):
            if path.name not in BILINEN_PIPELINE_GORSELLERI:
                return path
    return None


@st.cache_data
def load_hero_image_base64():
    config = Config()
    path = find_hero_image_file(config)
    if path is None:
        return None
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    uzanti = path.suffix.lstrip(".").lower()
    mime = "jpeg" if uzanti in ("jpg", "jpeg") else uzanti
    return f"data:image/{mime};base64,{encoded}"


def render_header():
    config = Config()
    hero_uri = load_hero_image_base64()
    metrikler = compute_overall_metrics_cached()

    if hero_uri:
        orta_gorsel = (
            f'<div style="position:relative; flex:1.4; min-width:220px; margin:0 20px; overflow:hidden;">'
            f'<img src="{hero_uri}" style="width:100%; height:100%; object-fit:cover; display:block;" />'
            f'<div style="position:absolute; inset:0; '
            f'background:linear-gradient(90deg, {BG_DARK} 0%, transparent 18%, transparent 82%, {BG_DARK} 100%), '
            f'linear-gradient(180deg, transparent 70%, {BG_DARK} 100%);"></div>'
            f'</div>'
        )
    else:
        orta_gorsel = f'<div style="flex:1; max-width:420px; margin:0 20px;">{pitch_network_svg()}</div>'

    html = f"""
    <div class="scout-header-wrap" style="align-items:stretch; flex-wrap:wrap;">
        <div style="display:flex; flex-direction:column; justify-content:center; gap:2px; flex-shrink:0;">
            <div class="scout-header-left">
                {hexagon_ball_logo_svg()}
                <div>
                    <div class="scout-title">SCOUT<span class="scout-title-white">AI</span></div>
                    <div class="scout-subtitle">FUTBOLCU SCOUTING, DEĞERLEME VE ÖNERİ PLATFORMU</div>
                </div>
            </div>
            {feature_chips_html()}
        </div>
        {orta_gorsel}
        <div style="padding-top:6px; flex-shrink:0;">
            {info_badges_html(metrikler["model_adi"])}
        </div>
    </div>
    """
    st.markdown(_flatten_html(html), unsafe_allow_html=True)




# Veri

@st.cache_data
def load_combined_data():
    config = Config()
    df = pd.read_csv(config.DATA_DIR / config.ML_READY_FILE)
    ref = pd.read_csv(config.DATA_DIR / config.PLAYER_REFERENCE_FILE)
    combined = pd.concat([ref.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined["_pozisyon"] = combined.apply(lambda r: get_position_display(r, config), axis=1)
    return combined


@st.cache_resource
def load_cost_model():
    config = Config()
    with open(config.DATA_DIR / config.BEST_MODEL_FILE, "rb") as f:
        return pickle.load(f)


@st.cache_data
def compute_overall_metrics_cached():
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    config = Config()
    cost_model_payload = load_cost_model()
    X_test = pd.read_csv(config.DATA_DIR / config.X_TEST_FILE)
    ref_test = pd.read_csv(config.DATA_DIR / config.REFERENCE_TEST_FILE)

    model = cost_model_payload["model"]
    is_log = cost_model_payload["is_log_model"]
    raw = model.predict(X_test)
    pred = np.expm1(raw) if is_log else raw
    pred = np.clip(pred, a_min=0, a_max=None)
    actual = ref_test[config.TARGET_COLUMN].values

    return {
        "model_adi": cost_model_payload["model_label"].split("_")[0],
        "r2": r2_score(actual, pred),
        "mae": mean_absolute_error(actual, pred),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
    }


@st.cache_data
def compute_price_segments_cached():
    config = Config()
    cost_model_payload = load_cost_model()
    X_test = pd.read_csv(config.DATA_DIR / config.X_TEST_FILE)
    ref_test = pd.read_csv(config.DATA_DIR / config.REFERENCE_TEST_FILE)

    model = cost_model_payload["model"]
    is_log = cost_model_payload["is_log_model"]
    raw_predictions = model.predict(X_test)
    predictions = np.expm1(raw_predictions) if is_log else raw_predictions
    predictions = np.clip(predictions, a_min=0, a_max=None)

    actual = ref_test[config.TARGET_COLUMN].values
    quantile_points = np.linspace(0, 1, config.NUM_PRICE_SEGMENTS + 1)
    edges = np.unique(np.quantile(actual, quantile_points))

    segments = []
    n = len(edges) - 1
    for i in range(n):
        lo = 0 if i == 0 else edges[i]
        hi = float("inf") if i == n - 1 else edges[i + 1]
        label = (f"{edges[i]:,.0f}-{edges[i+1]:,.0f} Euro" if i < n - 1 else f"{edges[i]:,.0f}+ Euro")
        segments.append({"lo": lo, "hi": hi, "label": label})

    df_pred = pd.DataFrame({"actual": actual, "predicted": predictions})
    df_pred["segment_label"] = df_pred["actual"].apply(lambda v: get_price_segment(v, segments)["label"])
    for seg in segments:
        rows = df_pred[df_pred["segment_label"] == seg["label"]]
        if len(rows) == 0:
            seg["mape"] = None
            continue
        valid = rows["actual"] > 0
        pct = np.abs(rows["predicted"][valid] - rows["actual"][valid]) / rows["actual"][valid] * 100
        seg["mape"] = float(pct.mean())

    return segments


def get_price_segment(value, segments):
    for seg in segments:
        if seg["lo"] <= value < seg["hi"]:
            return seg
    return segments[-1]


def get_position_group(row, config):
    for group_name, cols in config.POSITION_GROUPS.items():
        for col in cols:
            if col in row.index and row[col] == 1:
                return group_name
    return None


def get_position_display(row, config):
    for col, display_name in config.POSITION_DISPLAY.items():
        if col in row.index and row[col] == 1:
            return display_name
    return "Belirlenemedi"


def tr_country(name: str, config: Config) -> str:
    return config.COUNTRY_TR.get(name, name)


def _ilk_harf(isim: str) -> str:
    if not isinstance(isim, str) or not isim:
        return "#"
    sade = unicodedata.normalize("NFKD", isim[0]).encode("ascii", "ignore").decode("ascii")
    return sade.upper() if sade.isalpha() else "#"


@st.cache_data(show_spinner=False)
def mevcut_bas_harfler(isimler: tuple) -> list:
    return sorted({_ilk_harf(i) for i in isimler})


def player_selectbox(combined: pd.DataFrame, key: str, label: str = "Oyuncu Seçin:",
                     harf_filtresi: bool = False, pozisyon_filtresi: bool = False):
    havuz = combined

    if pozisyon_filtresi and "_pozisyon" in combined.columns:
        pozisyonlar = sorted(p for p in combined["_pozisyon"].dropna().unique()
                             if p != "Belirlenemedi")
        secilen_pozisyon = st.selectbox(
            "Pozisyon", ["Tüm Pozisyonlar"] + pozisyonlar, key=f"{key}_pozisyon",
        )
        if secilen_pozisyon != "Tüm Pozisyonlar":
            filtreli = havuz[havuz["_pozisyon"] == secilen_pozisyon]
            if len(filtreli) > 0:
                havuz = filtreli

    if harf_filtresi:
        harfler = mevcut_bas_harfler(tuple(havuz["player_name"].dropna().unique()))
        secilen_harf = st.radio(
            "Baş Harf", ["Tümü"] + harfler, key=f"{key}_harf", horizontal=True,
        )
        if secilen_harf != "Tümü":
            filtreli = havuz[havuz["player_name"].apply(_ilk_harf) == secilen_harf]
            if len(filtreli) > 0:
                havuz = filtreli

    etiketler = {}
    for idx, row in havuz[["player_name", "team_name", "season"]].iterrows():
        temel_etiket = f"{row['player_name']} — {row['team_name']} ({row['season']})"
        etiket = temel_etiket
        sayac = 2
        while etiket in etiketler:
            etiket = f"{temel_etiket} #{sayac}"
            sayac += 1
        etiketler[etiket] = idx
    secilen_etiket = st.selectbox(label, sorted(etiketler.keys()), key=key)
    return havuz.loc[etiketler[secilen_etiket]]


def team_selectbox(combined: pd.DataFrame, key: str, label: str = "Kulüp Seçin:",
                   harf_filtresi: bool = False, haric: str = None):
    takimlar = sorted(combined["team_name"].dropna().unique())
    if haric:
        takimlar = [t for t in takimlar if t != haric]

    if harf_filtresi:
        harfler = mevcut_bas_harfler(tuple(takimlar))
        secilen_harf = st.radio(
            "Baş Harf", ["Tümü"] + harfler, key=f"{key}_harf", horizontal=True,
        )
        if secilen_harf != "Tümü":
            filtreli = [t for t in takimlar if _ilk_harf(t) == secilen_harf]
            if filtreli:
                takimlar = filtreli

    return st.selectbox(label, takimlar, key=key)


BUTCE_SECENEKLERI = [
    500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000, 4_000_000,
    5_000_000, 6_000_000, 7_500_000, 10_000_000, 12_500_000, 15_000_000,
    20_000_000, 25_000_000, 30_000_000, 40_000_000, 50_000_000, 75_000_000,
    100_000_000, 150_000_000,
]


# Grafikler

def radar_chart(kategori_degerleri: dict):
    kategoriler = list(kategori_degerleri.keys())
    degerler = list(kategori_degerleri.values())
    kategoriler_kapali = kategoriler + [kategoriler[0]]
    degerler_kapali = degerler + [degerler[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=degerler_kapali, theta=kategoriler_kapali, fill="toself",
        line=dict(color=ACCENT_GREEN, width=2),
        fillcolor="rgba(164, 206, 57, 0.25)",
    ))
    layout = dict(PLOTLY_LAYOUT)
    layout.pop("bargap", None)
    layout.pop("bargroupgap", None)
    fig.update_layout(
        **layout,
        polar=dict(
            bgcolor=BG_CARD,
            radialaxis=dict(visible=True, range=[0, 100], color=TEXT_MUTED, gridcolor=BG_CARD_BORDER),
            angularaxis=dict(color=TEXT_LIGHT, gridcolor=BG_CARD_BORDER),
        ),
        showlegend=False,
        height=320,
    )
    return fig


def dual_bar_compare(oyuncu_deger, ortalama_deger, oyuncu_label="Oyuncu", ortalama_label="Ortalama"):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[""], x=[oyuncu_deger], name=oyuncu_label, orientation="h",
        marker_color=ACCENT_GREEN, text=[f"{oyuncu_deger:.0f}"], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        y=[""], x=[ortalama_deger], name=ortalama_label, orientation="h",
        marker_color=TEXT_MUTED, text=[f"{ortalama_deger:.0f}"], textposition="outside",
    ))
    layout = dict(PLOTLY_LAYOUT)
    layout["margin"] = dict(l=5, r=30, t=25, b=5)
    fig.update_layout(
        **layout, barmode="group", height=110,
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
    )
    return fig


def horizontal_skill_bars(row: pd.Series, feature_list: list, config: Config):
    labels = [config.DISPLAY_NAMES.get(f, f) for f in feature_list]
    values = [row[f] for f in feature_list]
    sirali = sorted(zip(labels, values), key=lambda x: x[1])
    labels_s, values_s = zip(*sirali)

    fig = go.Figure(go.Bar(
        x=values_s, y=labels_s, orientation="h",
        marker_color=ACCENT_GREEN, marker_line_color=BG_DARK, marker_line_width=1,
        text=[f"{v:.0f}" for v in values_s], textposition="outside",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=32 * len(feature_list) + 40,
        xaxis=dict(range=[0, 100], gridcolor=BG_CARD_BORDER, color=TEXT_MUTED),
        yaxis=dict(color=TEXT_LIGHT),
    )
    return fig


KORELASYON_ADAY_OZELLIKLERI = [
    "potential", "overall_rating", "Age", "matches_played", "goals", "assists",
    "attack_score", "midfield_score", "defense_score",
    "reactions", "vision", "short_passing", "ball_control", "long_passing",
    "dribbling", "curve", "volleys", "finishing", "shot_power", "long_shots",
    "crossing", "free_kick_accuracy", "penalties", "positioning",
    "acceleration", "sprint_speed", "agility", "balance", "stamina",
    "strength", "jumping", "aggression", "heading_accuracy",
    "interceptions", "marking", "standing_tackle", "sliding_tackle",
]


@st.cache_data(show_spinner=False)
def piyasa_degeri_korelasyonlari(combined: pd.DataFrame, hedef: str, adet: int = 15):
    adaylar = [c for c in KORELASYON_ADAY_OZELLIKLERI if c in combined.columns]
    kor = combined[adaylar + [hedef]].corr(numeric_only=True)[hedef].drop(hedef)
    kor = kor.dropna().sort_values(key=abs, ascending=False).head(adet)
    return kor.sort_values()


def korelasyon_figuru(kor: pd.Series, config: Config):
    labels = [config.DISPLAY_NAMES.get(i, i) for i in kor.index]
    renkler = [ACCENT_GREEN if v >= 0 else RED_PAHALI for v in kor.values]
    fig = go.Figure(go.Bar(
        x=kor.values, y=labels, orientation="h", marker_color=renkler,
        marker_line_color=BG_DARK, marker_line_width=1,
        text=[f"{v:+.2f}" for v in kor.values], textposition="outside",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=520,
        xaxis=dict(title="Korelasyon katsayısı", range=[-1, 1],
                   gridcolor=BG_CARD_BORDER, color=TEXT_MUTED, zerolinecolor=TEXT_MUTED),
        yaxis=dict(color=TEXT_LIGHT),
    )
    return fig


def deger_sacilim_figuru(combined: pd.DataFrame, config: Config):
    veri = combined[["overall_rating", config.TARGET_COLUMN, "Age", "player_name"]].dropna()
    veri = veri[veri[config.TARGET_COLUMN] > 0]
    fig = go.Figure(go.Scatter(
        x=veri["overall_rating"], y=veri[config.TARGET_COLUMN], mode="markers",
        marker=dict(size=5, color=veri["Age"], colorscale=[[0, ACCENT_GREEN], [1, ACCENT_PURPLE]],
                    opacity=0.65, showscale=True,
                    colorbar=dict(title="Yaş", thickness=12, outlinewidth=0)),
        text=veri["player_name"], hoverinfo="text+x+y",
    ))
    layout = dict(PLOTLY_LAYOUT)
    layout.pop("bargap", None)
    layout.pop("bargroupgap", None)
    fig.update_layout(
        **layout, height=420,
        xaxis=dict(title="Overall (Genel Seviye)", gridcolor=BG_CARD_BORDER, color=TEXT_MUTED),
        yaxis=dict(title="Piyasa Değeri (€)", type="log", tickformat="~s",
                   gridcolor=BG_CARD_BORDER, color=TEXT_MUTED),
    )
    return fig


def radar_chart_ikili(kategoriler: list, sol_degerler: list, sag_degerler: list,
                      sol_ad: str, sag_ad: str):
    kategoriler_kapali = kategoriler + [kategoriler[0]]
    fig = go.Figure()
    for degerler, ad, renk, dolgu in (
        (sol_degerler, sol_ad, ACCENT_BLUE, "rgba(79, 168, 224, 0.28)"),
        (sag_degerler, sag_ad, ACCENT_GREEN, "rgba(164, 206, 57, 0.28)"),
    ):
        fig.add_trace(go.Scatterpolar(
            r=degerler + [degerler[0]], theta=kategoriler_kapali, fill="toself",
            name=ad, line=dict(color=renk, width=2), fillcolor=dolgu,
        ))
    layout = dict(PLOTLY_LAYOUT)
    layout.pop("bargap", None)
    layout.pop("bargroupgap", None)
    fig.update_layout(
        **layout,
        polar=dict(
            bgcolor=BG_CARD,
            radialaxis=dict(visible=True, range=[0, 100], color=TEXT_MUTED, gridcolor=BG_CARD_BORDER),
            angularaxis=dict(color=TEXT_LIGHT, gridcolor=BG_CARD_BORDER),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, font=dict(size=11)),
        height=380,
    )
    return fig


def skill_bars_ikili(sol_row, sag_row, sol_ad: str, sag_ad: str,
                     feature_list: list, config: Config):
    labels = [config.DISPLAY_NAMES.get(f, f) for f in feature_list]
    sol_degerler = [float(sol_row[f]) for f in feature_list]
    sag_degerler = [float(sag_row[f]) for f in feature_list]

    sirali = sorted(zip(labels, sol_degerler, sag_degerler), key=lambda x: x[1])
    labels_s, sol_s, sag_s = zip(*sirali)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sol_s, y=labels_s, orientation="h", name=sol_ad,
        marker_color=ACCENT_BLUE, marker_line_color=BG_DARK, marker_line_width=1,
        text=[f"{v:.0f}" for v in sol_s], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=sag_s, y=labels_s, orientation="h", name=sag_ad,
        marker_color=ACCENT_GREEN, marker_line_color=BG_DARK, marker_line_width=1,
        text=[f"{v:.0f}" for v in sag_s], textposition="outside",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, barmode="group",
        height=42 * len(feature_list) + 60,
        xaxis=dict(range=[0, 112], gridcolor=BG_CARD_BORDER, color=TEXT_MUTED),
        yaxis=dict(color=TEXT_LIGHT),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, font=dict(size=11)),
    )
    return fig


def is_kaleci(row) -> bool:
    return "position_GK" in row.index and row["position_GK"] == 1


def radar_kategorileri_for(row, config: Config) -> dict:
    return config.RADAR_KATEGORILERI_KALECI if is_kaleci(row) else config.RADAR_KATEGORILERI


def panel_listeleri_for(row, config: Config) -> list:
    if is_kaleci(row):
        return [
            ("KALECİLİK", config.KALECILIK),
            ("FİZİKSELLİK", config.FIZIKSEL_HAREKET),
            ("TEKNİK", config.KALECI_AYAK_OYUNU),
        ]
    return [
        ("TEKNİK VE HÜCUM", config.TEKNIK_HUCUM),
        ("FİZİKSELLİK", config.FIZIKSEL_HAREKET),
        ("SAVUNMA", config.SAVUNMA),
    ]


def get_position_code(row, config: Config) -> str:
    for col in config.POSITION_DISPLAY:
        if col in row.index and row[col] == 1:
            return col.replace("position_", "")
    return "-"


def sentetik_profil_satiri(combined: pd.DataFrame, config: Config,
                           target_vector, ozellikler, position_group=None):
    tum_ozellikler = list(dict.fromkeys(
        list(config.SIMILARITY_FEATURES) + list(config.SIMILARITY_FEATURES_KALECI)))
    mevcut = [f for f in tum_ozellikler if f in combined.columns]
    satir = combined[mevcut].mean()

    for ad, deger in zip(ozellikler, target_vector):
        if ad in satir.index:
            satir[ad] = float(deger)

    for grup, kolonlar in config.POSITION_GROUPS.items():
        for kolon in kolonlar:
            satir[kolon] = 0
    if position_group and position_group in config.POSITION_GROUPS:
        satir[config.POSITION_GROUPS[position_group][0]] = 1

    satir["player_name"] = "Özel Profil"
    satir["team_name"] = "—"
    satir["season"] = "—"
    satir["League"] = "—"
    return satir


def ozel_profil_karti(row, config: Config, renk: str, position_group=None):
    grup_adi = (config.GROUP_DISPLAY_NAMES.get(position_group, position_group)
                if position_group else "Filtre yok")

    st.markdown(_flatten_html(f"""
    <div style="background-color:{BG_CARD}; border:1px solid {BG_CARD_BORDER};
                border-left:3px solid {renk}; border-radius:10px; padding:18px 16px;
                text-align:center; margin-bottom:10px;">
        <svg width="58" height="58" viewBox="0 0 40 40" style="margin-bottom:6px;">
            <circle cx="20" cy="20" r="16" fill="none" stroke="{renk}" stroke-width="2"/>
            <circle cx="20" cy="20" r="9" fill="none" stroke="{renk}" stroke-width="2"/>
            <circle cx="20" cy="20" r="2.5" fill="{renk}"/>
        </svg>
        <div style="font-size:19px; font-weight:800; color:{renk};">Özel Profil</div>
        <div style="color:{TEXT_MUTED}; font-size:12.5px; margin-top:6px; line-height:1.5;">
            Sizin belirlediğiniz hedef özellikler<br>
            Aranan pozisyon: <b style="color:{TEXT_LIGHT};">{grup_adi}</b>
        </div>
        <div style="color:{TEXT_MUTED}; font-size:11.5px; margin-top:10px;
                    border-top:1px solid {BG_CARD_BORDER}; padding-top:9px;">
            Bu profil veri setinde gerçek bir oyuncuya ait değildir;
            belirtilmeyen özellikler için veri seti ortalaması kullanılmıştır.
        </div>
    </div>
    """), unsafe_allow_html=True)


def player_compare_card(row, config: Config, renk: str):
    foto_url = get_player_photo_from_row(row)
    logo_url = get_team_logo_url(row["team_name"])
    pozisyon = get_position_display(row, config)
    kod = get_position_code(row, config)

    ust_col, bilgi_col = st.columns([1, 2])
    with ust_col:
        st.image(foto_url, width=110)
    with bilgi_col:
        st.markdown(_flatten_html(f"""
        <div style="font-size:20px; font-weight:800; color:{renk}; line-height:1.2;">
            {row['player_name']}
        </div>
        <div style="color:{TEXT_LIGHT}; font-size:14px; margin-top:6px; font-weight:600;">
            {row['team_name']}
        </div>
        <div style="color:{TEXT_MUTED}; font-size:12.5px; margin-top:2px;">
            {row['League']} · {row['season']}
        </div>
        <div style="color:{TEXT_MUTED}; font-size:12.5px; margin-top:6px;">
            Pozisyon: <b style="color:{TEXT_LIGHT};">{pozisyon} ({kod})</b>
            &nbsp;·&nbsp; Yaş: <b style="color:{TEXT_LIGHT};">{row['Age']:.0f}</b>
        </div>
        """), unsafe_allow_html=True)
        st.image(logo_url, width=34)

    m1, m2 = st.columns(2)
    m1.metric("Genel Seviye", f"{row['overall_rating']:.0f}")
    m2.metric("Potansiyel", f"{row['potential']:.0f}")
    st.metric("Piyasa Değeri", format_euro(row[config.TARGET_COLUMN]))


def player_comparison_block(sol_row, sag_row, config: Config,
                            sol_ozel_profil: bool = False, position_group=None):
    sol_ad = str(sol_row["player_name"])
    sag_ad = str(sag_row["player_name"])
    if sol_ad == sag_ad:
        sol_ad = f"{sol_ad} ({sol_row['team_name']})"
        sag_ad = f"{sag_ad} ({sag_row['team_name']})"

    ikisi_de_kaleci = is_kaleci(sol_row) and is_kaleci(sag_row)
    karisik = is_kaleci(sol_row) != is_kaleci(sag_row)

    kategori_sozlugu = (config.RADAR_KATEGORILERI_KALECI if ikisi_de_kaleci
                        else config.RADAR_KATEGORILERI)
    paneller = (panel_listeleri_for(sol_row, config) if ikisi_de_kaleci
                else [("TEKNİK VE HÜCUM", config.TEKNIK_HUCUM),
                      ("FİZİKSELLİK", config.FIZIKSEL_HAREKET),
                      ("SAVUNMA", config.SAVUNMA)])

    if karisik:
        st.caption("Karşılaştırılan oyunculardan yalnızca biri kaleci olduğu için "
                   "saha oyuncusu özellikleri gösteriliyor.")

    kol_sol, kol_orta, kol_sag = st.columns([1.1, 1.3, 1.1])

    with kol_sol:
        if sol_ozel_profil:
            ozel_profil_karti(sol_row, config, ACCENT_BLUE, position_group)
        else:
            player_compare_card(sol_row, config, ACCENT_BLUE)

    with kol_orta:
        kategoriler = list(kategori_sozlugu.keys())
        sol_degerler, sag_degerler = [], []
        for _, ozellikler in kategori_sozlugu.items():
            gecerli = [f for f in ozellikler if f in sol_row.index]
            sol_degerler.append(float(sol_row[gecerli].mean()))
            sag_degerler.append(float(sag_row[gecerli].mean()))
        render_chart(radar_chart_ikili(kategoriler, sol_degerler, sag_degerler, sol_ad, sag_ad))

    with kol_sag:
        player_compare_card(sag_row, config, ACCENT_GREEN)

    st.write("")
    kolonlar = st.columns(3)
    for kolon, (baslik, liste) in zip(kolonlar, paneller):
        with kolon:
            scout_card_start(baslik)
            render_chart(skill_bars_ikili(sol_row, sag_row, sol_ad, sag_ad, liste, config))
            scout_card_end()


def stat_cards_html(stats: list) -> str:
    cards = ""
    for icon_fn, label, value, color in stats:
        icon = icon_fn(color)
        cards += (
            f'<div class="scout-stat-card">'
            f'<svg width="26" height="26" viewBox="0 0 40 40">{icon}</svg>'
            f'<div><div class="scout-stat-label">{label}</div>'
            f'<div class="scout-stat-value" style="color:{color};">{value}</div></div>'
            f'</div>'
        )
    return f'<div class="scout-stat-row">{cards}</div>'


# Genel Bakış

def page_genel_bakis(combined: pd.DataFrame, config: Config):
    metrikler = compute_overall_metrics_cached()

    st.markdown(_flatten_html(stat_cards_html([
        (_icon_people, "OYUNCU", f"{combined['player_name'].nunique():,}", ACCENT_GREEN),
        (_icon_shield, "TAKIM", f"{combined['team_name'].nunique()}", ACCENT_GREEN),
        (_icon_globe, "LİG", f"{combined['League'].nunique()}", ACCENT_BLUE),
        (_icon_cpu, "EN İYİ MODEL", metrikler["model_adi"], ACCENT_PURPLE),
        (_icon_target, "R² SKORU", f"{metrikler['r2']:.2f}", ACCENT_GREEN),
        (_icon_ruler, "MAE", format_euro(metrikler["mae"]), ACCENT_BLUE),
        (_icon_chart_up, "RMSE", format_euro(metrikler["rmse"]), ACCENT_PURPLE),
    ])), unsafe_allow_html=True)

    sezonlar = sorted(combined["season"].dropna().unique())
    st.markdown(_flatten_html(f"""
    <div style="background-color:{BG_CARD}; border:1px solid {BG_CARD_BORDER};
                border-left:3px solid {ACCENT_BLUE}; border-radius:8px;
                padding:11px 16px; margin-bottom:16px;
                display:flex; align-items:center; gap:10px;">
        <svg width="18" height="18" viewBox="0 0 40 40">
            <circle cx="20" cy="20" r="16" fill="none" stroke="{ACCENT_BLUE}" stroke-width="2.5"/>
            <line x1="20" y1="17" x2="20" y2="29" stroke="{ACCENT_BLUE}" stroke-width="3"
                  stroke-linecap="round"/>
            <circle cx="20" cy="11.5" r="2" fill="{ACCENT_BLUE}"/>
        </svg>
        <div style="color:{TEXT_MUTED}; font-size:12.5px; line-height:1.5;">
            <b style="color:{TEXT_LIGHT};">Veri kapsamı:</b> Bu platformdaki tüm analizler
            <b style="color:{ACCENT_BLUE};">{sezonlar[0]}</b> ve
            <b style="color:{ACCENT_BLUE};">{sezonlar[-1]}</b> sezonlarına ait verilere dayanır.
            Her oyuncu için veri setindeki en güncel kayıt kullanılmıştır; piyasa değerleri
            o döneme aittir ve güncel transfer piyasasını yansıtmaz.
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown(_flatten_html(f"""
    <div class="scout-welcome-card">
        <div style="font-size:22px; font-weight:800; color:{TEXT_LIGHT};">
            ScoutAI'ya Hoş Geldiniz
        </div>
        <div style="color:{TEXT_MUTED}; margin-top:10px; font-size:13.5px; line-height:1.7;">
            Bu platform, gelişmiş makine öğrenmesi modelleri ve benzerlik algoritmaları kullanarak
            gizli yetenekleri keşfetmenize, piyasa değerlerini tahmin etmenize ve daha akıllı scout
            kararları almanıza yardımcı olur. Veriler European Soccer Database ve Transfermarkt
            referans alınarak hazırlanmıştır. Aşağıdaki üç modül, platformun temel yeteneklerini
            özetler: oyuncu ve takım verilerini keşfetmek, yapay zeka ile piyasa değeri tahmini
            yapmak ve istenen profile uygun benzer oyuncuları bulmak.
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.write("")

    fc1, fc2, fc3 = st.columns(3)
    kart_verileri = [
        (fc1, _icon_target, ACCENT_GREEN, "Yetenek Keşfi",
         "Gelişmiş filtrelerle binlerce oyuncuyu keşfedin, yetenek profillerini karşılaştırın."),
        (fc2, _icon_chart_up, ACCENT_PURPLE, "Değer Tahmini",
         "Yapay zeka destekli modeller, oyuncuların gerçek piyasa değerini yüksek doğrulukla tahmin eder."),
        (fc3, _icon_network, ACCENT_BLUE, "Akıllı Öneriler",
         "Kosinüs benzerlik motoruyla, hedef profilinize ve takım ihtiyacınıza uygun benzer oyuncuları keşfedin."),
    ]
    for col, icon_fn, renk, baslik, aciklama in kart_verileri:
        with col:
            icon_svg = icon_fn(renk)
            st.markdown(_flatten_html(f"""
            <div class="scout-feature-card" style="border-left-color:{renk};">
                <svg width="30" height="30" viewBox="0 0 40 40">{icon_svg}</svg>
                <div class="scout-feature-card-title">{baslik}</div>
                <div class="scout-feature-card-desc">{aciklama}</div>
            </div>
            """), unsafe_allow_html=True)


# Veri Analizi

def page_veri_analizi(combined: pd.DataFrame, config: Config):
    kor = piyasa_degeri_korelasyonlari(combined, config.TARGET_COLUMN)

    k1, k2 = st.columns([1.15, 1])

    with k1:
        scout_card_start("PİYASA DEĞERİNİ EN ÇOK ETKİLEYEN ÖZELLİKLER")
        st.markdown(
            f'<div style="color:{TEXT_MUTED}; font-size:12.5px; margin-bottom:6px;">'
            f'Her özelliğin piyasa değeri ile korelasyonu. Değer +1\'e yaklaştıkça özellik '
            f'arttığında piyasa değeri de artıyor, -1\'e yaklaştıkça ters yönde hareket ediyor.'
            f'</div>', unsafe_allow_html=True)
        render_chart(korelasyon_figuru(kor, config))

        en_guclu = kor.abs().idxmax()
        st.markdown(_flatten_html(f"""
        <div style="background-color:{BG_CARD}; border:1px solid {BG_CARD_BORDER};
                    border-left:3px solid {ACCENT_GREEN}; border-radius:8px;
                    padding:12px 14px; margin-top:8px;">
            <div style="color:{TEXT_MUTED}; font-size:11px; font-weight:700;
                        letter-spacing:0.4px;">EN GÜÇLÜ TEK ETKEN</div>
            <div style="color:{ACCENT_GREEN}; font-size:18px; font-weight:800; margin-top:3px;">
                {config.DISPLAY_NAMES.get(en_guclu, en_guclu)} ({kor[en_guclu]:+.2f})
            </div>
        </div>
        """), unsafe_allow_html=True)
        scout_card_end()

    with k2:
        scout_card_start("GENEL SEVİYE – PİYASA DEĞERİ İLİŞKİSİ")
        st.markdown(
            f'<div style="color:{TEXT_MUTED}; font-size:12.5px; margin-bottom:6px;">'
            f'Her nokta bir oyuncu, renk yaşı gösterir. Genel seviye yükseldikçe piyasa değeri '
            f'de yükseliyor; ancak aynı seviyedeki oyuncular arasında büyük değer farkları var. '
            f'İşte modelin devreye girdiği yer tam olarak bu boşluk.'
            f'</div>', unsafe_allow_html=True)
        render_chart(deger_sacilim_figuru(combined, config))
        scout_card_end()

    scout_card_start("LİGLERE GÖRE OYUNCU SAYISI")
    lig_sayilari = combined["League"].value_counts().head(11)
    fig = go.Figure(go.Bar(x=lig_sayilari.index, y=lig_sayilari.values, marker_color=ACCENT_GREEN,
                          marker_line_color=BG_DARK, marker_line_width=1,
                          text=lig_sayilari.values, textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, height=320,
                      xaxis=dict(color=TEXT_LIGHT, tickangle=-20),
                      yaxis=dict(gridcolor=BG_CARD_BORDER, color=TEXT_MUTED))
    render_chart(fig)
    scout_card_end()

    col1, col2 = st.columns(2)

    with col1:
        scout_card_start("POZİSYON DAĞILIMI")
        pos_sayilari = {}
        for grup, cols in config.POSITION_GROUPS.items():
            for c in cols:
                if c in combined.columns:
                    kisa_ad = c.replace("position_", "")
                    pos_sayilari[kisa_ad] = pos_sayilari.get(kisa_ad, 0) + int(combined[c].sum())
        pos_df = pd.Series(pos_sayilari).sort_values(ascending=True)
        fig = go.Figure(go.Bar(x=pos_df.values, y=pos_df.index, orientation="h", marker_color=ACCENT_GREEN,
                               marker_line_color=BG_DARK, marker_line_width=1,
                               text=pos_df.values, textposition="outside"))
        fig.update_layout(**PLOTLY_LAYOUT, height=350,
                          xaxis=dict(gridcolor=BG_CARD_BORDER, color=TEXT_MUTED),
                          yaxis=dict(color=TEXT_LIGHT))
        render_chart(fig)
        scout_card_end()

    with col2:
        scout_card_start("YAŞ DAĞILIMI")
        fig = go.Figure(go.Histogram(
            x=combined["Age"], marker_color=ACCENT_GREEN,
            marker_line_color=BG_DARK, marker_line_width=1.5, nbinsx=18,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=350,
                          xaxis=dict(title="Yaş", gridcolor=BG_CARD_BORDER, color=TEXT_MUTED),
                          yaxis=dict(gridcolor=BG_CARD_BORDER, color=TEXT_MUTED))
        render_chart(fig)
        scout_card_end()


# Oyuncu Analizi

def page_oyuncu_analizi(combined: pd.DataFrame, config: Config):
    oyuncu = player_selectbox(combined, key="oyuncu_analizi_secim", pozisyon_filtresi=True)

    pozisyon = get_position_display(oyuncu, config)
    foto_url = get_player_photo_from_row(oyuncu)
    logo_url = get_team_logo_url(oyuncu["team_name"])

    scout_card_start(f"{oyuncu['player_name'].upper()} — {oyuncu['team_name']} ({pozisyon})")
    foto_col, bilgi_col = st.columns([1, 5])
    with foto_col:
        if foto_url:
            st.image(foto_url, width=90)
        if logo_url:
            st.image(logo_url, width=36)
    with bilgi_col:
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.3, 1.8])
        c1.metric("Yaş", f"{oyuncu['Age']:.0f}")
        c2.metric("Genel Seviye", f"{oyuncu['overall_rating']:.0f}")
        c3.metric("Potansiyel", f"{oyuncu['potential']:.0f}")
        c4.metric("Piyasa Değeri", format_euro(oyuncu[config.TARGET_COLUMN]))
        c5.metric("Lig", oyuncu["League"])

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Maç Sayısı", f"{oyuncu['matches_played']:.0f}")
        s2.metric("Gol", f"{oyuncu['goals']:.0f}")
        s3.metric("Asist", f"{oyuncu['assists']:.0f}")
        s4.metric("Sarı Kart", f"{oyuncu['yellow_cards']:.0f}")
        s5.metric("Kırmızı Kart", f"{oyuncu['red_cards']:.0f}")
    scout_card_end()

    col1, col2 = st.columns([1, 1])

    with col1:
        scout_card_start("YETENEK PROFİLİ")
        kategori_sozlugu = radar_kategorileri_for(oyuncu, config)
        kategori_degerleri = {
            kategori: float(oyuncu[[f for f in ozellikler if f in oyuncu.index]].mean())
            for kategori, ozellikler in kategori_sozlugu.items()
        }
        render_chart(radar_chart(kategori_degerleri))
        scout_card_end()

    with col2:
        scout_card_start("LİG ORTALAMASIYLA KARŞILAŞTIRMA")
        lig_verisi = combined[combined["League"] == oyuncu["League"]]

        st.caption("Genel Seviye")
        render_chart(
            dual_bar_compare(oyuncu["overall_rating"], lig_verisi["overall_rating"].mean(),
                             "Oyuncu", "Lig Ortalaması"))
        st.caption("Potansiyel")
        render_chart(dual_bar_compare(oyuncu["potential"], lig_verisi["potential"].mean(),
                             "Oyuncu", "Lig Ortalaması"))
        scout_card_end()

    kolonlar = st.columns(3)
    for kolon, (baslik, liste) in zip(kolonlar, panel_listeleri_for(oyuncu, config)):
        with kolon:
            scout_card_start(baslik)
            render_chart(horizontal_skill_bars(oyuncu, liste, config))
            scout_card_end()


# Takım Analizi

def page_takim_analizi(combined: pd.DataFrame, config: Config):
    takim_listesi = sorted(combined["team_name"].dropna().unique())
    secilen_takim = st.selectbox("Takım Seçin:", takim_listesi, key="takim_analizi_secim")

    takim_satirlari = combined[combined["team_name"] == secilen_takim]
    profil = takim_satirlari[config.TEAM_TACTIC_FEATURES].mean()

    ornek_satir = takim_satirlari.iloc[0]
    logo_url = get_team_logo_url(secilen_takim)
    lig_logo_url = get_league_logo_url(ornek_satir["League"])
    bayrak_url = get_country_flag_url(ornek_satir["Country"])

    scout_card_start(secilen_takim.upper())
    logo_col, ulke_col, lig_col = st.columns([1, 2, 2])
    with logo_col:
        if logo_url:
            st.image(logo_url, width=76)
    with ulke_col:
        if bayrak_url:
            st.image(bayrak_url, width=34)
        st.markdown(_flatten_html(f"""
        <div style="color:{TEXT_MUTED}; font-size:11px; font-weight:600; letter-spacing:0.4px;">ÜLKE</div>
        <div style="color:{ACCENT_GREEN}; font-size:22px; font-weight:800;">
            {tr_country(ornek_satir["Country"], config)}
        </div>
        """), unsafe_allow_html=True)
    with lig_col:
        if lig_logo_url:
            st.image(lig_logo_url, width=34)
        st.markdown(_flatten_html(f"""
        <div style="color:{TEXT_MUTED}; font-size:11px; font-weight:600; letter-spacing:0.4px;">LİG</div>
        <div style="color:{ACCENT_GREEN}; font-size:22px; font-weight:800;">
            {ornek_satir["League"]}
        </div>
        """), unsafe_allow_html=True)
    scout_card_end()

    scout_card_start("TAKTİK PROFİLİ")
    labels = [config.DISPLAY_NAMES.get(c, c) for c in profil.index]
    sirali = sorted(zip(labels, profil.values), key=lambda x: x[1])
    labels_s, values_s = zip(*sirali)
    fig = go.Figure(go.Bar(x=values_s, y=labels_s, orientation="h", marker_color=ACCENT_GREEN,
                           marker_line_color=BG_DARK, marker_line_width=1,
                           text=[f"{v:.0f}" for v in values_s], textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, height=380,
                      xaxis=dict(range=[0, 112], gridcolor=BG_CARD_BORDER, color=TEXT_MUTED),
                      yaxis=dict(color=TEXT_LIGHT))
    render_chart(fig)
    scout_card_end()

    scout_card_start("TAKIM KADROSU")
    goster = takim_satirlari[["player_name", "Age", "overall_rating",
                              "potential", config.TARGET_COLUMN, "season"]].copy()
    goster.insert(1, "pozisyon", takim_satirlari.apply(lambda r: get_position_code(r, config), axis=1))
    goster[config.TARGET_COLUMN] = goster[config.TARGET_COLUMN].apply(format_euro)
    goster.columns = ["Oyuncu", "Pozisyon", "Yaş", "Genel Seviye", "Potansiyel",
                      "Piyasa Değeri", "Son Kayıtlı Sezon"]
    goster = goster.sort_values("Son Kayıtlı Sezon", ascending=False)
    st.dataframe(goster, use_container_width=True, hide_index=True)
    st.caption(
        "\"Son Kayıtlı Sezon\", her oyuncunun veri setindeki en güncel kaydının ait olduğu "
        "sezonu gösterir — bu takımın o sezonki tam kadrosu değildir. Bazı oyuncuların son "
        "kaydı 2014/2015'te kalmış olabilir; bu genellikle o oyuncunun sonraki sezonda veri "
        "kapsamı dışında kalan başka bir takıma/lige geçtiği anlamına gelir."
    )
    scout_card_end()


# Oyuncu Karşılaştırma

def page_oyuncu_karsilastirma(combined: pd.DataFrame, config: Config):
    scout_card_start("KARŞILAŞTIRILACAK OYUNCULAR")
    s1, s2 = st.columns(2)
    with s1:
        sol_row = player_selectbox(combined, key="karsilastirma_sol", label="1. Oyuncu:",
                                   pozisyon_filtresi=True)
    with s2:
        sag_row = player_selectbox(combined, key="karsilastirma_sag", label="2. Oyuncu:",
                                   pozisyon_filtresi=True)
    scout_card_end()

    player_comparison_block(sol_row, sag_row, config)


# Takım Karşılaştırma

TAKTIK_SINIF_GRUPLARI = {
    "OYUN KURULUM": [
        ("buildUpPlayPositioningClass_Organised", "Organize"),
        ("buildUpPlayPositioningClass_Free Form", "Serbest"),
    ],
    "HÜCUM": [
        ("chanceCreationPositioningClass_Organised", "Organize"),
        ("chanceCreationPositioningClass_Free Form", "Serbest"),
    ],
    "SAVUNMA": [
        ("defenceDefenderLineClass_Cover", "Kademeli"),
        ("defenceDefenderLineClass_Offside Trap", "Ofsayt Tuzağı"),
    ],
}


def taktik_sinif_etiketleri(satirlar: pd.DataFrame) -> dict:
    sonuc = {}
    for grup, secenekler in TAKTIK_SINIF_GRUPLARI.items():
        etiket = "-"
        for kolon, ad in secenekler:
            if kolon in satirlar.columns and satirlar[kolon].mean() >= 0.5:
                etiket = ad
                break
        sonuc[grup] = etiket
    return sonuc


TAKTIK_PANEL_YUKSEKLIK = 400


def taktik_profil_bar(profil: pd.Series, config: Config, renk: str, sira: list = None):
    sirali_index = sira if sira is not None else list(profil.index)
    degerler = [float(profil[c]) for c in sirali_index]
    labels = [config.DISPLAY_NAMES.get(c, c) for c in sirali_index]
    fig = go.Figure(go.Bar(
        x=degerler, y=labels, orientation="h", marker_color=renk,
        marker_line_color=BG_DARK, marker_line_width=1,
        text=[f"{v:.0f}" for v in degerler], textposition="outside",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=TAKTIK_PANEL_YUKSEKLIK,
                      xaxis=dict(range=[0, 112], gridcolor=BG_CARD_BORDER, color=TEXT_MUTED),
                      yaxis=dict(color=TEXT_LIGHT))
    return fig


def taktik_fark_bar(takim_profil: pd.Series, hedef_profil: pd.Series, config: Config,
                    sira: list = None):
    sirali_index = sira if sira is not None else list(takim_profil.index)
    fark = [round(float(takim_profil[c] - hedef_profil[c])) for c in sirali_index]
    labels = [config.DISPLAY_NAMES.get(c, c) for c in sirali_index]
    renkler = [ACCENT_GREEN if v >= 0 else RED_PAHALI for v in fark]
    fig = go.Figure(go.Bar(
        x=fark, y=labels, orientation="h", marker_color=renkler,
        marker_line_color=BG_DARK, marker_line_width=1,
        text=[f"{v:+.0f}" for v in fark], textposition="outside",
    ))
    sinir = max(12, max(abs(v) for v in fark) * 1.6)
    fig.update_layout(**PLOTLY_LAYOUT, height=TAKTIK_PANEL_YUKSEKLIK,
                      xaxis=dict(range=[-sinir, sinir], gridcolor=BG_CARD_BORDER,
                                 color=TEXT_MUTED, zerolinecolor=TEXT_MUTED),
                      yaxis=dict(color=TEXT_LIGHT))
    return fig


def sinif_kartlari_html(etiketler: dict, renk: str) -> str:
    kutular = "".join(
        f'<div style="flex:1; background-color:{BG_CARD}; border:1px solid {BG_CARD_BORDER}; '
        f'border-radius:8px; padding:12px 10px; text-align:center;">'
        f'<div style="color:{TEXT_MUTED}; font-size:10.5px; font-weight:700; letter-spacing:0.4px;">{grup}</div>'
        f'<div style="color:{renk}; font-size:16px; font-weight:800; margin-top:5px;">{deger}</div>'
        f'</div>'
        for grup, deger in etiketler.items()
    )
    return f'<div style="display:flex; gap:10px; margin-top:12px;">{kutular}</div>'


def page_takim_karsilastirma(combined: pd.DataFrame, config: Config):
    takim_listesi = sorted(combined["team_name"].dropna().unique())

    scout_card_start("KARŞILAŞTIRMA AYARLARI")
    s1, s2 = st.columns(2)
    with s1:
        secilen_takim = st.selectbox("Takım Seçin:", takim_listesi, key="tk_takim")
    with s2:
        hedef_tipi = st.radio(
            "Karşılaştırma Hedefi",
            ["Lig Ortalaması", "Başka Bir Takım"],
            key="tk_hedef_tipi", horizontal=True,
        )
        rakip_takim = None
        if hedef_tipi == "Başka Bir Takım":
            diger_takimlar = [t for t in takim_listesi if t != secilen_takim]
            rakip_takim = st.selectbox("Rakip Takım:", diger_takimlar, key="tk_rakip")
    scout_card_end()

    takim_satirlari = combined[combined["team_name"] == secilen_takim]
    takim_profil = takim_satirlari[config.TEAM_TACTIC_FEATURES].mean()
    takim_siniflar = taktik_sinif_etiketleri(takim_satirlari)

    if hedef_tipi == "Lig Ortalaması":
        lig = takim_satirlari.iloc[0]["League"]
        hedef_satirlar = combined[combined["League"] == lig]
        hedef_ad = f"{lig} (Ortalama)"
    else:
        hedef_satirlar = combined[combined["team_name"] == rakip_takim]
        hedef_ad = rakip_takim

    hedef_profil = hedef_satirlar[config.TEAM_TACTIC_FEATURES].mean()
    hedef_siniflar = taktik_sinif_etiketleri(hedef_satirlar)

    sol_logo = get_team_logo_url(secilen_takim)
    sag_logo = get_team_logo_url(rakip_takim) if rakip_takim else get_league_logo_url(
        takim_satirlari.iloc[0]["League"])

    ortak_sira = list(takim_profil.sort_values().index)

    def panel_ust_bilgi(logo_url, baslik, renk):
        logo_html = (f'<img src="{logo_url}" style="max-height:44px; max-width:64px; '
                     f'object-fit:contain;" />' if logo_url else "")
        return _flatten_html(f"""
        <div style="height:52px; display:flex; align-items:center;">{logo_html}</div>
        <div style="height:30px; display:flex; align-items:center; color:{renk};
                    font-weight:800; font-size:15px;">{baslik}</div>
        """)

    k1, k2, k3 = st.columns(3)

    with k1:
        scout_card_start("TAKIM ÖZELLİKLERİ")
        st.markdown(panel_ust_bilgi(sol_logo, secilen_takim, ACCENT_GREEN),
                    unsafe_allow_html=True)
        render_chart(taktik_profil_bar(takim_profil, config, ACCENT_GREEN, ortak_sira))
        st.markdown(_flatten_html(sinif_kartlari_html(takim_siniflar, ACCENT_GREEN)),
                    unsafe_allow_html=True)
        scout_card_end()

    with k2:
        scout_card_start("KARŞILAŞTIRMA HEDEFİ")
        st.markdown(panel_ust_bilgi(sag_logo, hedef_ad, ACCENT_BLUE),
                    unsafe_allow_html=True)
        render_chart(taktik_profil_bar(hedef_profil, config, ACCENT_BLUE, ortak_sira))
        st.markdown(_flatten_html(sinif_kartlari_html(hedef_siniflar, ACCENT_BLUE)),
                    unsafe_allow_html=True)
        scout_card_end()

    with k3:
        scout_card_start("FARK ANALİZİ")
        st.markdown(panel_ust_bilgi(None,
                                    f"Yeşil: {secilen_takim} daha yüksek", TEXT_MUTED),
                    unsafe_allow_html=True)
        render_chart(taktik_fark_bar(takim_profil, hedef_profil, config, ortak_sira))
        scout_card_end()


# Karar Masası

def compute_similarity(combined, target_vector, config, ozellikler=None):
    kullanilacak = ozellikler if ozellikler is not None else config.SIMILARITY_FEATURES
    X = combined[kullanilacak].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    target_scaled = scaler.transform(target_vector.reshape(1, -1))
    return pd.Series(cosine_similarity(target_scaled, X_scaled)[0], index=combined.index)


def compute_team_fit_raw(combined, team_style, config):
    normalized_style = {k: v / 100.0 for k, v in team_style.items()}
    skill_matrix = combined[config.SIMILARITY_FEATURES].copy() / 100.0
    weighted_scores = pd.Series(0.0, index=combined.index)
    total_weight = 0.0
    for tactic_col, skill_cols in config.TACTIC_TO_SKILL_MAP.items():
        weight = normalized_style.get(tactic_col, 0.5)
        for skill_col in skill_cols:
            if skill_col in skill_matrix.columns:
                weighted_scores += weight * skill_matrix[skill_col]
                total_weight += weight
    if total_weight > 0:
        weighted_scores /= total_weight
    return weighted_scores


def classify_value_gap(actual, predicted, segments):
    if actual <= 0:
        return "BİLİNMİYOR"
    seg = get_price_segment(actual, segments)
    mape = seg.get("mape")
    if mape is None or mape <= 0:
        return "BİLİNMİYOR"
    fark_yuzde = (predicted - actual) / actual * 100
    if fark_yuzde > mape:
        return "FIRSAT"
    elif fark_yuzde < -mape:
        return "PAHALI"
    return "MAKUL"


def get_target_vector_ui(combined: pd.DataFrame, config: Config):
    mod = st.radio(
        "Oyuncu Profili Kaynağı",
        ["Bir Yıldıza Benzet", "Özel Profil Oluştur"],
        key="oyuncu_modu", horizontal=True,
    )

    if mod == "Bir Yıldıza Benzet":
        eslesen = player_selectbox(combined, key="hedef_oyuncu_secim",
                                   label="Örnek Alınacak Oyuncu:", harf_filtresi=True,
                                   pozisyon_filtresi=True)
        ozellikler = (config.SIMILARITY_FEATURES_KALECI if is_kaleci(eslesen)
                      else config.SIMILARITY_FEATURES)
        target_vector = eslesen[ozellikler].values.astype(float)
        target_label = f"{eslesen['player_name']} ({eslesen['team_name']}, {eslesen['season']})"
        excluded_index = eslesen.name
        position_group = get_position_group(eslesen, config)
        pozisyon_gosterim = get_position_display(eslesen, config)
        st.caption(f"Tespit edilen pozisyon: **{pozisyon_gosterim}**")
        if is_kaleci(eslesen):
            st.caption("Kaleci profili tespit edildi — benzerlik, kalecilik "
                       "özellikleri üzerinden hesaplanacak.")
        return target_vector, target_label, excluded_index, position_group, ozellikler

    else:
        pozisyon_secenekleri = ["(Filtre Yok)"] + list(config.POSITION_GROUPS.keys())
        secilen_pozisyon = st.selectbox(
            "Aranacak Pozisyon:", pozisyon_secenekleri, key="manuel_pozisyon",
            format_func=lambda k: config.GROUP_DISPLAY_NAMES.get(k, k) if k != "(Filtre Yok)" else k,
        )
        position_group = None if secilen_pozisyon == "(Filtre Yok)" else secilen_pozisyon
        kaleci_modu = secilen_pozisyon == "KALECI"
        ozellikler = config.SIMILARITY_FEATURES_KALECI if kaleci_modu else config.SIMILARITY_FEATURES

        ortalama = combined[ozellikler].mean()
        target_vector = ortalama.copy()
        with st.expander("Özellikleri Ayarla (0-100)", expanded=True):
            onemli_ozellikler = (
                ["gk_reflexes", "gk_diving", "gk_handling", "gk_positioning",
                 "gk_kicking", "reactions", "jumping", "short_passing"]
                if kaleci_modu else
                ["sprint_speed", "short_passing", "dribbling", "interceptions",
                 "stamina", "finishing", "vision", "standing_tackle"]
            )
            cols = st.columns(2)
            for i, ozellik in enumerate(onemli_ozellikler):
                with cols[i % 2]:
                    deger = st.slider(config.DISPLAY_NAMES.get(ozellik, ozellik), 0, 100,
                                      int(ortalama[ozellik]), key=f"manuel_{ozellik}")
                    target_vector[ozellik] = deger
        return target_vector.values.astype(float), "Özel Profil", None, position_group, ozellikler


def get_team_style_ui(combined: pd.DataFrame, config: Config):
    mod = st.radio(
        "Hedef Kulüp",
        ["Kulüp Seç", "Taktik Profilini Elle Gir"],
        key="takim_modu", horizontal=True,
    )

    if mod == "Kulüp Seç":
        secilen_takim = team_selectbox(combined, key="hedef_takim_secim",
                                       label="Kulüp Seçin:", harf_filtresi=True)
        takim_satirlari = combined[combined["team_name"] == secilen_takim]
        team_style = {col: takim_satirlari[col].mean() for col in config.TEAM_TACTIC_FEATURES}
        return team_style, secilen_takim
    else:
        team_style = {}
        with st.expander("Kulübün Taktik Profilini Ayarlayın", expanded=True):
            cols = st.columns(3)
            for i, ozellik in enumerate(config.TEAM_TACTIC_FEATURES):
                with cols[i % 3]:
                    team_style[ozellik] = st.slider(config.DISPLAY_NAMES.get(ozellik, ozellik),
                                                    0, 100, 50, key=f"takim_manuel_{ozellik}")
        return team_style, "Özel Kulüp Profili"


def scout_raporu_hesapla(combined, config, target_vector, team_style, excluded_index,
                         position_group, max_budget, top_n, ozellikler=None):
    gelisim_ivmesi = combined["potential"] - combined["overall_rating"]
    gelisim_gecti = gelisim_ivmesi >= 0

    benzerlik = compute_similarity(combined, target_vector, config, ozellikler)
    takim_uyum_ham = compute_team_fit_raw(combined, team_style, config)

    result = combined[["player_api_id", "player_name", "team_name", "season", "Age",
                       "overall_rating", "potential", config.TARGET_COLUMN]].copy()
    result["pozisyon_grubu"] = combined.apply(lambda row: get_position_group(row, config), axis=1)
    result["pozisyon_gosterim"] = combined.apply(lambda row: get_position_display(row, config), axis=1)
    result["benzerlik_skoru"] = benzerlik
    result["takim_uyum_skoru_ham"] = takim_uyum_ham
    result["gelisim_filtresi_gecti"] = gelisim_gecti

    if excluded_index is not None:
        result = result.drop(index=excluded_index)
    if position_group is not None:
        result = result[result["pozisyon_grubu"] == position_group]
    result = result[result["gelisim_filtresi_gecti"]]
    if max_budget is not None:
        result = result[result[config.TARGET_COLUMN] <= max_budget]

    if len(result) == 0:
        return None

    if result["takim_uyum_skoru_ham"].max() > result["takim_uyum_skoru_ham"].min():
        result["takim_uyum_skoru"] = (
            (result["takim_uyum_skoru_ham"] - result["takim_uyum_skoru_ham"].min())
            / (result["takim_uyum_skoru_ham"].max() - result["takim_uyum_skoru_ham"].min())
        )
    else:
        result["takim_uyum_skoru"] = 0.5

    result["scout_skoru"] = (
        config.BENZERLIK_AGIRLIGI * result["benzerlik_skoru"]
        + config.TAKIM_UYUM_AGIRLIGI * result["takim_uyum_skoru"]
    )
    result = result.sort_values("scout_skoru", ascending=False).head(top_n)

    cost_model_payload = load_cost_model()
    segments = compute_price_segments_cached()
    X_train_columns = pd.read_csv(config.DATA_DIR / config.X_TRAIN_FILE, nrows=0).columns.tolist()
    candidate_features = combined.loc[result.index, X_train_columns]

    model = cost_model_payload["model"]
    is_log = cost_model_payload["is_log_model"]
    raw_pred = model.predict(candidate_features)
    pred = np.expm1(raw_pred) if is_log else raw_pred
    pred = np.clip(pred, a_min=0, a_max=None)
    result["motor1_adil_deger"] = pred
    result["maliyet_yorumu"] = [
        classify_value_gap(a, p, segments)
        for a, p in zip(result[config.TARGET_COLUMN], result["motor1_adil_deger"])
    ]
    return result


def oneri_karti_html(row, config: Config, renk: str) -> str:
    foto_url = get_player_photo_from_row(row)
    foto_html = (
        f'<img src="{foto_url}" style="width:38px;height:38px;border-radius:50%;'
        f'object-fit:cover;vertical-align:middle;margin-right:10px;'
        f'border:1px solid {BG_CARD_BORDER};" />'
    )
    return f"""
    <div class="scout-card" style="border-left: 4px solid {renk}; margin-bottom:6px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="display:flex; align-items:center;">
                {foto_html}
                <div>
                    <span style="font-size:17px; font-weight:700; color:{TEXT_LIGHT};">{row['player_name']}</span>
                    <span style="color:{TEXT_MUTED};"> — {row['team_name']} ({row['season']}) · {row['pozisyon_gosterim']}</span>
                </div>
            </div>
            <div style="background-color:{renk}; color:#0B0E14; font-weight:700; padding:4px 14px; border-radius:6px;">
                {row['maliyet_yorumu']}
            </div>
        </div>
        <div style="margin-top:10px; color:{TEXT_MUTED}; font-size:13px;">
            Yaş: <b style="color:{TEXT_LIGHT};">{row['Age']:.0f}</b> &nbsp;|&nbsp;
            Benzerlik: <b style="color:{ACCENT_GREEN};">%{row['benzerlik_skoru']*100:.1f}</b> &nbsp;|&nbsp;
            Kulüp Uyumu: <b style="color:{ACCENT_GREEN};">%{row['takim_uyum_skoru']*100:.1f}</b> &nbsp;|&nbsp;
            Scout Skoru: <b style="color:{ACCENT_GREEN};">%{row['scout_skoru']*100:.1f}</b>
        </div>
        <div style="margin-top:6px; color:{TEXT_LIGHT}; font-size:13px;">
            Transfermarkt Değeri: <b>{format_euro(row[config.TARGET_COLUMN])}</b> &nbsp;|&nbsp;
            Model Tahmini: <b>{format_euro(row['motor1_adil_deger'])}</b>
        </div>
    </div>
    """


def page_akilli_karar_masasi(combined: pd.DataFrame, config: Config):
    scout_card_start("ARAMA KRİTERLERİ")
    target_vector, target_label, excluded_index, position_group, ozellikler = get_target_vector_ui(combined, config)
    st.write("")
    team_style, team_label = get_team_style_ui(combined, config)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Öneri Sayısı", 3, 20, 10)
    with col2:
        butce_aktif = st.checkbox("Transfer Bütçesi Belirle")
        max_budget = None
        if butce_aktif:
            max_budget = st.selectbox(
                "Üst Sınır:", BUTCE_SECENEKLERI,
                index=BUTCE_SECENEKLERI.index(10_000_000),
                format_func=format_euro_tam, key="butce_secim",
            )
    scout_card_end()

    if st.button("SCOUT RAPORU OLUŞTUR", type="primary", use_container_width=True):
        with st.spinner("Hesaplanıyor..."):
            result = scout_raporu_hesapla(combined, config, target_vector, team_style,
                                          excluded_index, position_group, max_budget,
                                          top_n, ozellikler)
        if result is None:
            st.session_state.pop("scout_result", None)
            st.warning("Belirlenen kriterlere uyan oyuncu bulunamadı.")
            return
        st.session_state["scout_result"] = result
        st.session_state["scout_target_index"] = excluded_index
        st.session_state["scout_target_label"] = target_label
        st.session_state["scout_team_label"] = team_label
        st.session_state["scout_target_vector"] = target_vector
        st.session_state["scout_ozellikler"] = ozellikler
        st.session_state["scout_position_group"] = position_group
        st.session_state.pop("scout_karsilastir", None)

    result = st.session_state.get("scout_result")
    if result is None:
        return

    st.success(f"{len(result)} aday bulundu — Hedef: {st.session_state['scout_target_label']}"
               f"  |  Kulüp: {st.session_state['scout_team_label']}")

    hedef_index = st.session_state.get("scout_target_index")
    ozel_profil_modu = hedef_index is None
    st.caption("Bir adayı hedef profille yan yana karşılaştırmak için satırındaki "
               "**Karşılaştır** düğmesine basın.")

    renk_haritasi = {"FIRSAT": ACCENT_GREEN, "PAHALI": RED_PAHALI, "MAKUL": TEXT_MUTED}

    secilen_idx = st.session_state.get("scout_karsilastir")

    for idx, row in result.iterrows():
        renk = renk_haritasi.get(row["maliyet_yorumu"], TEXT_MUTED)
        kart_col, dugme_col = st.columns([9, 1.6])
        with kart_col:
            st.markdown(_flatten_html(oneri_karti_html(row, config, renk)), unsafe_allow_html=True)
        with dugme_col:
            etiket = "Kapat" if secilen_idx == idx else "Karşılaştır"
            if st.button(etiket, key=f"cmp_{idx}", use_container_width=True):
                if secilen_idx == idx:
                    st.session_state.pop("scout_karsilastir", None)
                else:
                    st.session_state["scout_karsilastir"] = idx
                st.rerun()

        if secilen_idx == idx:
            st.markdown(f'<div class="scout-section-title">HEDEF PROFİL vs '
                        f'{row["player_name"].upper()}</div>', unsafe_allow_html=True)
            if ozel_profil_modu:
                hedef_satir = sentetik_profil_satiri(
                    combined, config,
                    st.session_state.get("scout_target_vector"),
                    st.session_state.get("scout_ozellikler"),
                    st.session_state.get("scout_position_group"),
                )
                player_comparison_block(
                    hedef_satir, combined.loc[idx], config,
                    sol_ozel_profil=True,
                    position_group=st.session_state.get("scout_position_group"),
                )
            else:
                player_comparison_block(combined.loc[hedef_index], combined.loc[idx], config)
            st.markdown('<div class="scout-section-end"></div>', unsafe_allow_html=True)


# Ana Uygulama

def main():
    st.set_page_config(page_title="ScoutAI - Akıllı Scout Sistemi", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(_flatten_html(f"""
    <div class="scout-mobile-menu-hint">
        <span style="color:{ACCENT_GREEN}; font-size:18px; font-weight:900;">&raquo;</span>
        <span>Sayfalar arasında geçiş için sol üstteki
        <b style="color:{ACCENT_GREEN};">&raquo;</b> simgesine dokunun</span>
    </div>
    """), unsafe_allow_html=True)

    render_header()

    config = Config()
    combined = load_combined_data()

    with st.sidebar:
        st.markdown(_flatten_html(f"""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:18px;">
            {hexagon_ball_logo_svg(38)}
            <div style="font-weight:800; font-size:16px; color:{ACCENT_GREEN};">SCOUT<span style="color:{TEXT_LIGHT};">AI</span></div>
        </div>
        """), unsafe_allow_html=True)

        sayfa = st.radio(
            "Menü",
            [
                "🏠 Genel Bakış",
                "👤 Oyuncu Analizi",
                "👥 Oyuncu Karşılaştırma",
                "🛡️ Takım Analizi",
                "⚖️ Takım Karşılaştırma",
                "📊 Veri Analizi",
                "🎯 Transfer Önerileri",
            ],
            label_visibility="collapsed",
        )

    if sayfa == "🏠 Genel Bakış":
        page_genel_bakis(combined, config)
    elif sayfa == "👤 Oyuncu Analizi":
        page_oyuncu_analizi(combined, config)
    elif sayfa == "👥 Oyuncu Karşılaştırma":
        page_oyuncu_karsilastirma(combined, config)
    elif sayfa == "🛡️ Takım Analizi":
        page_takim_analizi(combined, config)
    elif sayfa == "⚖️ Takım Karşılaştırma":
        page_takim_karsilastirma(combined, config)
    elif sayfa == "📊 Veri Analizi":
        page_veri_analizi(combined, config)
    else:
        page_akilli_karar_masasi(combined, config)


if __name__ == "__main__":
    main()
