from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import urllib.request
import ssl
import gzip
from bs4 import BeautifulSoup
import yfinance as yf


ROOT = Path(__file__).resolve().parent
OUTPUT_CSV = ROOT / "clean_recovery_supernova_labeled_dataset.csv"
OUTPUT_JSON = ROOT / "clean_recovery_supernova_build_meta.json"
NY_TZ = ZoneInfo("America/New_York")
BUILD_DATE = date.today().isoformat()
HISTORY_START = "2023-01-01"
EVENT_START = pd.Timestamp("2024-01-01", tz=NY_TZ)
TODAY = pd.Timestamp.now(tz=NY_TZ).normalize()
RECENT_INTRADAY_CUTOFF = TODAY - pd.Timedelta(days=30)

QUOTAS = OrderedDict(
    [
        ("TOXIC_EXCLUDE", 15),
        ("OPEN_DRIVE_SQUEEZE", 10),
        ("CLEAN_RECOVERY_2X", 25),
        ("NEAR_POSITIVE", 25),
        ("FAILED_SETUP", 25),
    ]
)

MANUAL_SEEDS = [
    "MYSE",
    "WSHP",
    "WNW",
    "CTNT",
    "BIRD",
    "HUBC",
    "ASTI",
    "SOPA",
    "WATT",
    "MNTS",
    "EVTV",
    "AIXI",
    "BNAI",
    "SHAZ",
    "QUBT",
    "RGTI",
    "QBTS",
    "KULR",
    "RVSN",
    "OPTT",
    "GOVX",
    "CERO",
    "NRXP",
    "DATS",
    "KITT",
    "LTRY",
    "GNS",
    "SERV",
    "RCAT",
    "AGRI",
    "EVAX",
    "SNOA",
    "ICU",
    "ENVB",
    "TOVX",
    "CMND",
    "PBM",
    "FGI",
    "JLHL",
    "CNEY",
    "BYND",
    "CMPS",
    "CUE",
    "IMMP",
    "FFIE",
    "MULN",
    "SYTA",
    "SNTI",
    "WISA",
    "LUCY",
    "TOP",
    "XHG",
    "AGAE",
    "VSA",
]

MANUAL_LABEL_HINTS = {
    "WNW": "TOXIC_EXCLUDE",
