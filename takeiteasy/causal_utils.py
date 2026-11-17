"""Shared prompt-construction helpers for the CausalLM (Qwen/Llama) adaptation
of the TakeItEasy pipeline.

The important subtlety here is truncation. PubHealth premises routinely exceed
the context window, and the naive approach -- tokenize the whole chat-formatted
string with truncation=True -- silently destroys the example:

  * in training, the truncation point lands before the assistant turn, so every
    label is -100 and the example contributes exactly zero loss;
  * in inference, the trailing generation prompt is cut off, so the model is
    never actually asked to answer.

Both are silent: training still "succeeds" and reports a loss of 0. So instead
of truncating the finished prompt, we truncate the *user body text* and leave the
chat scaffolding and the answer intact.
"""

SYSTEM_MSG = "You are a fact verification model. Return exactly one label."

# Small slack so that decode()/re-encode() round-tripping cannot push us back
# over the limit.
_MARGIN = 8


def build_chat_messages(input_string, answer_string=None):
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": input_string},
    ]
    if answer_string is not None:
        messages.append({"role": "assistant", "content": answer_string})
    return messages


def _encode(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _render(tokenizer, input_string, answer_string):
    """Return (prompt_ids, answer_ids) for the given body text."""
    prompt_text = tokenizer.apply_chat_template(
        build_chat_messages(input_string, None),
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = _encode(tokenizer, prompt_text)

    if answer_string is None:
        return prompt_ids, []

    full_text = tokenizer.apply_chat_template(
        build_chat_messages(input_string, answer_string),
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = _encode(tokenizer, full_text)
    # apply_chat_template is prefix-stable for both Qwen and Llama-3.1 chat
    # templates (verified), so the answer is exactly the tail.
    return prompt_ids, full_ids[len(prompt_ids):]


def fit_body_to_budget(tokenizer, input_string, answer_string, max_length):
    """Shrink the user body until prompt+answer fits in max_length.

    Returns (prompt_ids, answer_ids). The answer is never truncated.
    """
    prompt_ids, answer_ids = _render(tokenizer, input_string, answer_string)
    if len(prompt_ids) + len(answer_ids) <= max_length:
        return prompt_ids, answer_ids

    body_ids = _encode(tokenizer, input_string)
    # Tokens the chat template itself contributes, independent of body length.
    overhead = len(prompt_ids) - len(body_ids)
    budget = max_length - len(answer_ids) - overhead - _MARGIN
    if budget < 1:
        budget = 1

    shortened = tokenizer.decode(body_ids[:budget], skip_special_tokens=True)
    prompt_ids, answer_ids = _render(tokenizer, shortened, answer_string)

    # Final hard clamp. Drop from the LEFT so the tail -- end of the user turn
    # plus the generation prompt -- always survives.
    overflow = len(prompt_ids) + len(answer_ids) - max_length
    if overflow > 0:
        prompt_ids = prompt_ids[overflow:]

    return prompt_ids, answer_ids


def tokenize_for_causal_lm(tokenizer, input_string, answer_string, max_length):
    """Build (input_ids, labels, attention_mask) with the prompt masked out."""
    prompt_ids, answer_ids = fit_body_to_budget(
        tokenizer, input_string, answer_string, max_length
    )

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)

    if not any(l != -100 for l in labels):
        raise ValueError(
            "no supervised tokens -- the answer was truncated away "
            f"(prompt={len(prompt_ids)}, answer={len(answer_ids)}, max={max_length})"
        )
    return input_ids, labels, attention_mask


def build_inference_prompt_ids(tokenizer, input_string, max_length, reserve=8):
    """Prompt token ids for generation, with the generation prompt preserved."""
    prompt_ids, _ = fit_body_to_budget(
        tokenizer, input_string, None, max_length - reserve
    )
    return prompt_ids
