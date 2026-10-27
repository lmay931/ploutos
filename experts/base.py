"""
Base classes for the experts.

An Expert holds everything one expert does in the pipeline: its features, the
prompt block it adds, its directional call, and the metadata the generator/
teacher/ eval need. Adding an expert should just be a new file (subclass +
register) and an entry in config.active_experts.

Labels: experts store their own native labels (BUY/SELL/NEUTRAL) and that's
what gets printed in the prompt and teacher blocks. The eval and
experts_correct work with normalised UP/DOWN/NEUTRAL/None, which signal()
gets from the native label via `vocabulary`. An expert that uses UP/DOWN
natively can just override `vocabulary`.

Data scope: build_features(panel, config) sees the whole universe and returns
columns keyed by (Ticker, Date), which get merged into the master panel. So by
the time make_input()/ signal() run on a row everything is already a column,
and the per-row methods don't care about scope:
  - single ticker: group by Ticker (own history)
  - cross-sectional: group by Date (rank vs peers)
  - external: load your own series (rates/ oil/ FX) and map onto (Ticker, Date)
"""
from __future__ import annotations

from typing import Literal, Optional, Protocol, Tuple, runtime_checkable

import pandas as pd

# normalised labels used by the eval/ experts_correct
UP = "UP"
DOWN = "DOWN"
NEUTRAL = "NEUTRAL"

DataScope = Literal["single_ticker", "cross_sectional", "external"]


class ResponseSchemaError(Exception):
    """Raised by LLMExpert.parse_response when the LLM response doesn't match
    the schema. The generator logs these as schema failures."""


@runtime_checkable
class Expert(Protocol):
    """Interface every expert follows. DeterministicExpert/ LLMExpert below
    do the shared bits."""

    id: str                  # stable registry key, e.g. "pm" / "technical"
    name: str                # human label, e.g. "Price & Macro"
    source_tag: str          # teacher rationale tag, e.g. "PRICE_MACRO"
    prompt_title: str        # teacher block header stem, e.g. "Price & Macro Expert"
    signal_column: str       # master-CSV column holding the native signal
    input_column: str        # master-CSV column holding the prompt input block
    view_column: Optional[str]
    include_basis: bool      # teacher block prints "Basis: {view}"
    data_scope: DataScope
    is_llm: bool
    persona: Optional[str]   # system prompt, only for LLM experts

    def build_features(self, panel: pd.DataFrame, config) -> pd.DataFrame: ...
    def make_input(self, row) -> str: ...
    def signal(self, row) -> Optional[str]: ...


class BaseExpert:
    """Shared attributes and label normalisation. Subclass DeterministicExpert
    or LLMExpert rather than this."""

    # identity/ wiring, override in subclasses
    id: str = ""
    name: str = ""
    source_tag: str = ""
    prompt_title: str = ""
    signal_column: str = ""
    input_column: str = ""
    view_column: Optional[str] = None
    include_basis: bool = False
    data_scope: DataScope = "single_ticker"
    is_llm: bool = False
    persona: Optional[str] = None

    # whether the expert can abstain with NEUTRAL. the teacher keeps these rows
    # (fills NEUTRAL) rather than dropping them, and doesn't make up a rationale
    # for an abstaining expert
    can_abstain: bool = False
    # one-liner for the "source expert" list in the teacher persona
    teacher_description: str = ""

    # normalised direction -> native label (what's stored and shown in prompts).
    # the inverse maps stored labels back to UP/DOWN
    vocabulary = {UP: "BUY", DOWN: "SELL", NEUTRAL: "NEUTRAL"}

    # ------------------------------------------------------------------
    def _normalize(self, native_label) -> Optional[str]:
        """Map a native label (e.g. 'BUY') to UP/DOWN/NEUTRAL.

        Missing, NaN or unknown labels give None. Labels that are already
        UP/DOWN/NEUTRAL are passed through as is.
        """
        if native_label is None:
            return None
        if isinstance(native_label, float) and pd.isna(native_label):
            return None
        if native_label in (UP, DOWN, NEUTRAL):
            return native_label
        inverse = {v: k for k, v in self.vocabulary.items()}
        return inverse.get(native_label)

    # ------------------------------------------------------------------
    # subclasses implement these
    def build_features(self, panel: pd.DataFrame, config) -> pd.DataFrame:
        raise NotImplementedError

    def make_input(self, row) -> str:
        raise NotImplementedError

    def signal(self, row) -> Optional[str]:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "LLM" if self.is_llm else "deterministic"
        return f"<{type(self).__name__} id={self.id!r} ({kind})>"


class DeterministicExpert(BaseExpert):
    """Expert whose signal is just a function of the features, no LLM call.

    Subclasses implement compute(row) -> (native_label, view) and make_input.
    signal() recomputes from the features, so a new deterministic expert can be
    scored on the test set without any LLM/ GPU cost - useful for checking it
    actually beats 50% before distilling.
    """

    is_llm = False
    persona = None

    def compute(self, row) -> Tuple[str, str]:
        """Return (native_label, view) for this row."""
        raise NotImplementedError

    def signal(self, row) -> Optional[str]:
        native, _ = self.compute(row)
        return self._normalize(native)


class LLMExpert(BaseExpert):
    """Expert that calls an LLM (Gemini) with a persona.

    Subclasses provide persona, schema, make_input and parse_response(result),
    which turns the LLM's JSON into (native_label, view, agg_or_None).
    signal() reads the cached signal_column from generation time, so the eval
    doesn't need to call the API again.
    """

    is_llm = True
    persona: Optional[str] = None
    schema: Optional[dict] = None
    # True if the LLM returns scores that parse_response turns into a signal
    # in code (sentiment), False if the LLM gives the signal directly (fundamental)
    aggregates: bool = False

    def parse_response(self, result: dict) -> Tuple[str, str, Optional[float]]:
        """Map an LLM JSON response to (native_label, view, agg_or_None)."""
        raise NotImplementedError

    def signal(self, row) -> Optional[str]:
        return self._normalize(row.get(self.signal_column))
