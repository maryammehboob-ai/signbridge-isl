"""HTML building blocks and the stylesheet loader for the SignBridge look (presentation only)."""
import functools
import html
from pathlib import Path

import streamlit as st

CSS_PATH = Path(__file__).resolve().parent / "style.css"
BRAND = "SignBridge"
TAGLINE = "Indian Sign Language, in both directions"

# Original line icons (24x24 grid), drawn for this app.
ICONS = {
    "home": "<path d='M3 11l9-7 9 7'/><path d='M5 10v10h14V10'/><path d='M10 20v-6h4v6'/>",
    "camera": "<rect x='3' y='6' width='14' height='12' rx='3'/><path d='M17 10l4-2v8l-4-2'/><circle cx='10' cy='12' r='2.2'/>",
    "signs": "<rect x='3' y='4' width='7' height='7' rx='2'/><rect x='14' y='4' width='7' height='7' rx='2'/><rect x='3' y='14' width='7' height='7' rx='2'/><path d='M14 17.5h7M17.5 14v7'/>",
    "mic": "<rect x='9' y='3' width='6' height='11' rx='3'/><path d='M5 11a7 7 0 0 0 14 0M12 18v3M9 21h6'/>",
}


@functools.lru_cache(maxsize=1)
def _css():
    return CSS_PATH.read_text(encoding="utf-8")


def inject():
    """Apply the stylesheet to the page (call once at the top of the app and of every page)."""
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


def icon(name, size=None):
    body = ICONS.get(name, ICONS["home"])
    style = f" style='width:{size}px;height:{size}px'" if size else ""
    return f"<svg viewBox='0 0 24 24'{style}>{body}</svg>"


def chip(text, kind="", live=False):
    """A small status pill. kind: '' | ok | warn | bad."""
    css = " ".join(part for part in ("sb-chip", kind, "live" if live else "") if part)
    return f"<span class='{css}'><span class='dot'></span>{html.escape(str(text))}</span>"


def page_header(title, subtitle, icon_name="home", eyebrow="", chips=()):
    """The top of a page: a small icon label, the title (a real Streamlit title), a one-line description, optional chips."""
    st.markdown(f"<div class='sb-eyebrow'><span class='ico'>{icon(icon_name)}</span>{html.escape(eyebrow or BRAND)}</div>",
                unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)
    if chips:
        st.markdown(f"<div class='sb-chips'>{''.join(chips)}</div>", unsafe_allow_html=True)


def eyebrow(icon_name, text):
    """Just the small icon label (for pages whose own title is drawn by the page itself)."""
    st.markdown(f"<div class='sb-eyebrow'><span class='ico'>{icon(icon_name)}</span>{html.escape(text)}</div>", unsafe_allow_html=True)


def empty_video(message="The camera is off. Click <b>Start Camera</b>."):
    """The placeholder shown where the live video goes while the camera is off."""
    return f"<div class='sb-empty'><div>{icon('camera')}<b>Camera off</b>{message}</div></div>"


def status_check(ok, label):
    mark = "<span class='mark ok'>&#10003;</span>" if ok else "<span class='mark bad'>!</span>"
    return f"<div class='sb-check'>{mark}<span>{html.escape(label)}</span></div>"


def footer():
    st.markdown(f"<div class='sb-footer'>{BRAND} &middot; fingerspelling for Indian Sign Language (A&ndash;Z, 1&ndash;9) "
                f"&middot; live camera pictures are analysed on the server and never stored; speech-to-text uses Google.</div>", unsafe_allow_html=True)


def side_footer():
    st.sidebar.markdown("<div class='sb-side-foot'><b>Tips</b><br>Allow the camera when your browser asks. "
                        "Keep both hands inside the picture for two-hand signs.</div>", unsafe_allow_html=True)
