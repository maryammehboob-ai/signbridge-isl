"""Helpers for the Text-to-Sign page: sign pictures embedded in the page, accent clean-up, GIF.

Everything is fingerspelling: every word is shown letter by letter, because the project has pictures
only for the 35 signs A-Z and 1-9 (no whole-word signs exist, so none can be shown).

The reference pictures are embedded in the page as data: pictures. A normal Streamlit image goes through
Streamlit's image server, which failed earlier (HTTP 500 with starlette 1.x); an embedded picture is part
of the page itself, so it can never be broken by the server.
"""
import base64
import functools
import io
import unicodedata

from core import PROJECT_DIR

from text_to_sign import IMAGE_DIR, SPACE_LABEL, displayable, split_words

CARD = 96                       # picture size in the grid (pixels)
MAX_GIF_SIGNS = 60              # a GIF longer than this would be too big to show


def normalize_text(text):
    """Letters with accents become plain letters (e-acute -> E) so they can be signed.
    Returns (new text, list of messages about what was changed)."""
    out, notes = [], []
    for char in text or "":
        if ord(char) < 128:
            out.append(char)
            continue
        plain = "".join(c for c in unicodedata.normalize("NFKD", char) if not unicodedata.combining(c))
        plain = plain if plain.isascii() and plain else ""
        if plain and plain.isalpha():
            out.append(plain)
            notes.append(f"'{char}' is shown as {plain.upper()}")
        else:
            out.append(char)                                   # left for text_to_tokens to report
    seen, unique = set(), []
    for note in notes:
        if note not in seen:
            seen.add(note)
            unique.append(note)
    return "".join(out), unique


@functools.lru_cache(maxsize=64)
def data_uri(path):
    """A reference picture as a data: URI (cached)."""
    return "data:image/jpeg;base64," + base64.b64encode(open(path, "rb").read()).decode("ascii")


def _card(token, size=CARD, highlight=False):
    """One sign (or space, or a missing picture) as a card. Presentation only; the style lives in ui/style.css."""
    box = f"width:{size}px;height:{size}px"
    if token.kind == "space":
        return f"<div class='sg-space' style='{box}'>{SPACE_LABEL}</div>"
    if token.image is None:
        return f"<div class='sg-missing' style='{box}'>no picture<br>for {token.char}</div>"
    current = " current" if highlight else ""
    return (f"<div class='sg-card{current}' style='width:{size}px'>"
            f"<img src='{data_uri(str(token.image))}' alt='sign {token.char}' style='{box}'>"
            f"<span class='ch'>{token.char}</span></div>")


def sequence_html(tokens, size=CARD):
    """All signs, word by word, as one block of HTML."""
    words = split_words(tokens)
    parts = []
    for number, word in enumerate(words, start=1):
        cards = "".join(_card(t, size) for t in word)
        parts.append(f"<div class='sg-word'><span class='n'>Word {number}:</span> <span class='w'>{''.join(t.char for t in word)}</span></div>"
                     f"<div class='sg-row'>{cards}</div>")
        if number < len(words):
            parts.append(f"<div class='sg-divider'>{SPACE_LABEL}</div>")
    return "<div>" + "".join(parts) + "</div>"


def single_html(token, size=300):
    return "<div class='sg-row'>" + _card(token, size, highlight=True) + "</div>"


def make_gif(tokens, seconds_per_sign=0.9, size=320):
    """An animated GIF that shows the signs one after another. Returns bytes (None if nothing to show)."""
    from PIL import Image, ImageDraw, ImageFont

    steps = displayable(tokens)
    if not steps or len(steps) > MAX_GIF_SIGNS:
        return None
    try:
        font = ImageFont.truetype("arial.ttf", 28)
        small = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = small = ImageFont.load_default()
    frames = []
    for number, token in enumerate(steps, start=1):
        canvas = Image.new("RGB", (size, size + 70), (20, 22, 30))
        draw = ImageDraw.Draw(canvas)
        if token.kind == "space":
            draw.rectangle([20, 20, size - 20, size - 20], outline=(150, 150, 150), width=3)
            draw.text((size // 2, size // 2), SPACE_LABEL, fill=(190, 190, 190), font=font, anchor="mm")
            label = SPACE_LABEL
        elif token.image is not None:
            picture = Image.open(token.image).convert("RGB").resize((size, size))
            canvas.paste(picture, (0, 0))
            label = token.char
        else:
            draw.text((size // 2, size // 2), f"no picture for {token.char}", fill=(231, 76, 60), font=small, anchor="mm")
            label = token.char
        draw.text((size // 2, size + 24), label, fill=(255, 255, 255), font=font, anchor="mm")
        draw.text((size // 2, size + 52), f"{number} of {len(steps)}", fill=(150, 150, 150), font=small, anchor="mm")
        frames.append(canvas)
    buffer = io.BytesIO()
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:], duration=int(seconds_per_sign * 1000),
                   loop=0, optimize=False)
    return buffer.getvalue()


def gif_html(gif_bytes, width=320):
    uri = "data:image/gif;base64," + base64.b64encode(gif_bytes).decode("ascii")
    return f"<div class='sg-gif'><img src='{uri}' alt='animated signs' style='width:{width}px'></div>"
