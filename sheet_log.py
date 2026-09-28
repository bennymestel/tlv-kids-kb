"""Log searches, ratings and feedback to a Google Sheet in the background.

Needs st.secrets["gcp_service_account"] and st.secrets["sheet_id"]; without
them (e.g. running locally with no secrets.toml) logging is silently skipped.
"""
import sys
import threading
from datetime import datetime, timezone

import gspread
import streamlit as st


@st.cache_resource
def _sheet():
    if "gcp_service_account" not in st.secrets or "sheet_id" not in st.secrets:
        return None
    try:
        client = gspread.service_account_from_dict(dict(st.secrets["gcp_service_account"]))
        return client.open_by_key(st.secrets["sheet_id"])
    except Exception as e:
        print(f"sheet_log: could not open sheet: {e}", file=sys.stderr)
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(tab: str, row: list):
    sheet = _sheet()
    if sheet is None:
        return

    def work():
        try:
            sheet.worksheet(tab).append_row(row)
        except Exception as e:
            print(f"sheet_log: failed to append to {tab}: {e}", file=sys.stderr)

    threading.Thread(target=work, daemon=True).start()


def log_search(query: str, results: list):
    top = [qa["question"] for qa in results[:3]]
    top += [""] * (3 - len(top))
    _append("searches", [_now(), query, len(results), *top])


def log_rating(query: str, thumbs: str, results: list):
    # Denormalized (query + what was shown) so the ratings tab is readable on its
    # own, without cross-referencing the searches tab.
    top = [qa["question"] for qa in results[:3]]
    top += [""] * (3 - len(top))
    _append("ratings", [_now(), query, thumbs, *top])


def log_feedback(text: str):
    _append("feedback", [_now(), text])
