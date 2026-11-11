"""The look of the app (presentation only): the stylesheet and a few HTML building blocks.

Nothing here touches the models, the camera, the recognition, the speech or the sign pictures. Pages import
this module to draw headers, chips and cards; all behaviour stays in APP/core and in the page callbacks.
"""
from ui.theme import chip, empty_video, eyebrow, footer, inject, page_header, side_footer, status_check  # noqa: F401
