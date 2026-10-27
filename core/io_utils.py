# helpers for reading the JSONL datasets and turning rows into token ids

import json


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def to_messages(row):
    """Turn a JSONL row's messages into the format the HF chat template wants
    (renames role 'model' to 'assistant')."""
    out = []
    for m in row["messages"]:
        role = "assistant" if m["role"] == "model" else m["role"]
        out.append({"role": role, "content": m["content"]})
    return out


def encode_chat(msgs, tokenizer, add_generation_prompt):
    """Render a message list to token ids.

    Renders to a string with apply_chat_template(tokenize=False) first and then
    tokenises it. Passing the messages straight to the tokenizer gives back an
    Encoding object rather than a list, which quietly breaks the prompt/
    response boundary used for masking.
    """
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def render_and_mask(row, tokenizer, max_len):
    """Return (input_ids, labels) with prompt tokens masked to -100."""
    msgs = to_messages(row)
    full_ids = encode_chat(msgs, tokenizer, add_generation_prompt=False)
    prompt_ids = encode_chat(msgs[:-1], tokenizer, add_generation_prompt=True)
    boundary = len(prompt_ids)
    labels = list(full_ids)
    for i in range(min(boundary, len(labels))):
        labels[i] = -100
    return full_ids[:max_len], labels[:max_len]
