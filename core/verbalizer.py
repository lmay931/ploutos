"""
UP/DOWN word lists and the direction extractor.

There are two separate sets of words here:

DTW_UP_WORDS/ DTW_DOWN_WORDS are the 4-word anchors from Ploutos Appendix C,
used to build the type embedding t_Y_d for the DTW loss. Don't add to these -
the point is that t_UP and t_DOWN end up pointing in clearly opposite
directions, and adding looser synonyms pulls both towards the middle of
"directional words" and weakens the contrast.

EXTRACT_UP_WORDS/ EXTRACT_DOWN_WORDS are much broader and only used by
extract_direction(), for checking teacher outputs and for scoring student
predictions at eval. These should cover whatever wording a model might use
("appreciation", "downside", "rally", "tumble" etc.), so add to them whenever
a case gets missed.

extract_direction just scans the first ~25 word tokens and returns the first
directional word it finds, without splitting into sentences. I tried splitting
on [.!?] first but tickers/ names have full stops in them ("D.R. Horton",
"A.O. Smith"), so the first "sentence" would just be "D" and nothing was found.
The teacher and student both give the verdict in the opening clause anyway, and
the justification (which can mention the opposite direction) comes after the
first 25 tokens.

Negation: a few tokens before the directional word (NEG_WINDOW) are checked for
a negator, e.g. "unlikely to decline" -> UP, "little downside" -> UP, "limited
upside" -> DOWN. Only that word gets flipped, not the whole sentence. It handles
the common cases but it's not a proper negation parser, so it will get some
awkward multi-clause sentences wrong - worth spot checking on bigger runs.
"""

import re

# ---------------------------------------------------------------------------
# DTW anchors, straight from Ploutos Appendix C
# ---------------------------------------------------------------------------
DTW_UP_WORDS = ["up", "boost", "positive", "rise"]
DTW_DOWN_WORDS = ["down", "reduce", "negative", "drop"]


# ---------------------------------------------------------------------------
# extraction word lists, used to parse the prediction text
# ---------------------------------------------------------------------------
EXTRACT_UP_WORDS = frozenset({
    "up", "upward", "upwards", "upside",
    "rise", "rises", "rising", "risen",
    "appreciate", "appreciates", "appreciating", "appreciation",
    "gain", "gains", "gaining",
    "increase", "increases", "increasing",
    "rally", "rallies", "rallying",
    "climb", "climbs", "climbing",
    "advance", "advances", "advancing",
    "surge", "surges", "surging",
    "bullish", "positive", "higher",
    "rebound", "rebounds", "rebounding",
    "recover", "recovers", "recovering", "recovery",
    "boost", "boosts", "boosting",
})

EXTRACT_DOWN_WORDS = frozenset({
    "down", "downward", "downwards", "downside",
    "drop", "drops", "dropping",
    "decline", "declines", "declining",
    "fall", "falls", "falling", "fell",
    "decrease", "decreases", "decreasing",
    "tumble", "tumbles", "tumbling",
    "slide", "slides", "sliding",
    "plunge", "plunges", "plunging",
    "weaken", "weakens", "weakening",
    "bearish", "negative", "lower",
    "retreat", "retreats", "retreating",
    "depreciate", "depreciates", "depreciating", "depreciation",
    "reduce", "reduces", "reducing", "reduction",
    "loss", "losses",
})


# ---------------------------------------------------------------------------
# negation for the extractor
#
# a negator in the NEG_WINDOW tokens before a directional word flips it.
# "little"/ "limited" are weaker than "not"/ "never" and might misfire (e.g.
# "limited gains" is arguably still a small UP), so drop them if that happens.
# NEG_WINDOW has to stay small - a wider window or flipping the whole sentence
# breaks hedged predictions like "rise, though some decline is possible" -> UP
# ---------------------------------------------------------------------------
NEGATORS = frozenset({
    "not", "no", "never", "unlikely", "without", "hardly", "scarcely",
    "fails", "fail", "failing", "failed", "isnt", "wont", "doesnt", "cannot",
    "little", "limited",
})
NEG_WINDOW = 3

# how many leading word tokens to scan - enough to get past the ticker and the
# verdict clause, but stops before the justification that mentions the other side
SCAN_CAP = 25


def extract_direction(prediction_text, fallback_to_full=False):
    """
    Get UP/DOWN (or None) from a prediction paragraph.

    Returns the first directional word in the first SCAN_CAP tokens, after
    negation. If an up and a down word both show up (hedged predictions) the
    earlier one wins, since the main claim usually comes before the qualifier:
    "X will rise, though some decline is possible" -> UP.

    A leading "Prediction:" is stripped. With fallback_to_full=True it scans
    the whole text if nothing is found early on - useful at eval for messy
    student outputs, but I leave it off when checking the teacher so openers
    with no direction get flagged.
    """
    if not prediction_text or not prediction_text.strip():
        return None

    cleaned = re.sub(r'^\s*Prediction:\s*', '', prediction_text,
                     count=1, flags=re.IGNORECASE)

    result = _scan_first_tokens(cleaned, cap=SCAN_CAP)
    if result is not None:
        return result

    if fallback_to_full:
        return _scan_first_tokens(cleaned, cap=10 ** 9)

    return None


def _scan_first_tokens(text, cap=SCAN_CAP):
    """First directional word in the first `cap` tokens (with negation), or None."""
    if not text:
        return None

    tokens = re.findall(r"[a-z]+", text.lower())[:cap]
    for i, t in enumerate(tokens):
        is_up = t in EXTRACT_UP_WORDS
        is_down = t in EXTRACT_DOWN_WORDS
        if not (is_up or is_down):
            continue

        window = tokens[max(0, i - NEG_WINDOW):i]
        negated = any(w in NEGATORS for w in window)

        direction = "UP" if is_up else "DOWN"
        if negated:
            direction = "DOWN" if direction == "UP" else "UP"
        return direction

    return None


# ---------------------------------------------------------------------------
# token ids for the DTW type embedding, used by train/sft_train.py
# ---------------------------------------------------------------------------
def get_verbalizer_token_ids(label, tokenizer, prepend_space=True):
    """
    Token ids of the DTW anchor words for label ("UP" or "DOWN"), taking the
    first sub-token of each word.

    With BPE/ SentencePiece tokenizers (Llama, Mistral, Qwen) a word mid-sentence
    tokenises differently to one at the start, so a space is prepended to get
    the mid-sentence token, which is what shows up in the actual responses.
    Only set prepend_space=False if you've checked your tokenizer doesn't do this.
    """
    if label not in ("UP", "DOWN"):
        raise ValueError(f"label must be 'UP' or 'DOWN', got {label!r}")

    words = DTW_UP_WORDS if label == "UP" else DTW_DOWN_WORDS
    token_ids = []
    for w in words:
        text = (" " + w) if prepend_space else w
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise RuntimeError(f"Tokenizer returned no IDs for {text!r}")
        token_ids.append(ids[0])
    return token_ids
