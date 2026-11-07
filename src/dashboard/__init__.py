"""Streamlit dashboard package (``src.dashboard.state`` = data layer, ``src.dashboard.app`` = UI).

This module also hosts :func:`container_width_kwargs`, a tiny compatibility
adapter for the full-width keyword of Streamlit commands. It deliberately does
not import ``streamlit`` (it only inspects a callable's signature) so the pure
data layer and the tests can import the package without a Streamlit runtime.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Final

__all__ = ["LEGACY_FULL_WIDTH", "MODERN_FULL_WIDTH", "container_width_kwargs"]

#: Keyword form understood by every Streamlit release in the declared range (``>=1.38``).
LEGACY_FULL_WIDTH: Final[dict[str, object]] = {"use_container_width": True}
#: Keyword form of recent releases, where ``use_container_width`` is deprecated (and later removed).
MODERN_FULL_WIDTH: Final[dict[str, object]] = {"width": "stretch"}


def container_width_kwargs(command: Callable[..., object]) -> dict[str, object]:
    """Keyword arguments that stretch a Streamlit element to its container.

    Streamlit renamed the full-width switch of ``st.button``,
    ``st.download_button``, ``st.dataframe`` and ``st.plotly_chart`` from
    ``use_container_width=True`` to ``width="stretch"``; the old keyword is
    deprecated and scheduled for removal, while the new one does not exist in
    older releases (there ``width`` is a pixel integer). The project declares
    ``streamlit>=1.38``, so the right form is picked at runtime from the
    command's signature:

    * ``use_container_width`` present with a boolean default -> old API,
      :data:`LEGACY_FULL_WIDTH`;
    * ``use_container_width`` absent or defaulting to ``None`` (the deprecated
      tri-state that forwards to ``width``) -> new API, :data:`MODERN_FULL_WIDTH`.

    Callables whose signature cannot be inspected get the legacy form, which is
    valid on every release of the declared range. The returned mapping is a
    fresh copy, safe to mutate or to splat into the call.
    """
    try:
        parameters = inspect.signature(command).parameters
    except (TypeError, ValueError):
        return dict(LEGACY_FULL_WIDTH)
    legacy = parameters.get("use_container_width")
    if legacy is not None and isinstance(legacy.default, bool):
        return dict(LEGACY_FULL_WIDTH)
    return dict(MODERN_FULL_WIDTH)
