"""Lightweight token estimation for context management.

Provides fast, dependency-free token counting that works well for
Chinese/English mixed text. No need for tiktoken or model-specific tokenizers.

Approximation rules (based on empirical testing with Claude/GPT tokenizers):
- English words: ~1.3 tokens per word
- CJK characters: ~1.5 tokens per character (most CJK chars = 1-2 tokens)
- Punctuation/symbols: ~1 token each
- Numbers: ~0.5 tokens per digit cluster

This is intentionally conservative (slightly overestimates) to prevent
context overflow. For exact counts, use tiktoken.
"""

from __future__ import annotations

import re

# Regex to classify character types
_CJK_RE = re.compile(
    r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff'
    r'\U00020000-\U0002a6df\U0002a700-\U0002b73f'
    r'\u3000-\u303f\uff00-\uffef]'
)
_WORD_RE = re.compile(r'[a-zA-Z]+')
_NUM_RE = re.compile(r'\d+')


def estimate_tokens(text: str) -> int:
    """Estimate token count for mixed Chinese/English text.

    Returns a conservative (slightly high) estimate.
    """
    if not text:
        return 0

    tokens = 0

    # Count CJK characters (~1.5 tokens each)
    cjk_count = len(_CJK_RE.findall(text))
    tokens += int(cjk_count * 1.5)

    # Remove CJK for remaining analysis
    remaining = _CJK_RE.sub(' ', text)

    # Count English words (~1.3 tokens each)
    words = _WORD_RE.findall(remaining)
    tokens += int(len(words) * 1.3)

    # Count number clusters (~1 token per 3 digits)
    nums = _NUM_RE.findall(remaining)
    for n in nums:
        tokens += max(1, len(n) // 3 + 1)

    # Remaining punctuation/symbols (~1 token each significant one)
    stripped = _WORD_RE.sub('', _NUM_RE.sub('', remaining))
    significant = sum(1 for c in stripped if not c.isspace())
    tokens += significant

    return max(tokens, 1)


def truncate_to_tokens(text: str, max_tokens: int, *, tail: bool = False) -> str:
    """Truncate text to approximately max_tokens.

    Args:
        text: Input text.
        max_tokens: Maximum token budget.
        tail: If True, keep the tail (last N tokens) instead of head.

    Returns:
        Truncated text with indicator if truncated.
    """
    if estimate_tokens(text) <= max_tokens:
        return text

    # Binary search for the right cut point
    lines = text.split('\n')

    if tail:
        lines = list(reversed(lines))

    kept = []
    total = 0
    for line in lines:
        line_tokens = estimate_tokens(line)
        if total + line_tokens > max_tokens:
            # Try to fit a partial line
            if not kept:
                # At least keep something from the first line
                chars_budget = int(max_tokens * 2.5)  # rough chars-per-token
                kept.append(line[:chars_budget])
            break
        kept.append(line)
        total += line_tokens + 1  # +1 for newline token

    if tail:
        kept = list(reversed(kept))

    result = '\n'.join(kept)

    original_tokens = estimate_tokens(text)
    if len(kept) < len(lines):
        pos = "末尾" if not tail else "开头"
        result += f"\n... (truncated, ~{original_tokens} tokens total, kept ~{max_tokens})"

    return result


def truncate_middle(text: str, max_tokens: int) -> str:
    """Keep head and tail, truncate middle.

    Useful for shell output where exit code is at top and
    recent output is at bottom.
    """
    if estimate_tokens(text) <= max_tokens:
        return text

    half = max_tokens // 2
    head = truncate_to_tokens(text, half)
    tail = truncate_to_tokens(text, half, tail=True)

    # Remove the truncation indicators from head/tail before combining
    head_clean = head.rsplit('\n...', 1)[0] if '\n...' in head else head
    tail_clean = tail.split('...', 1)[-1].lstrip('\n') if tail.startswith('...') else tail

    total_est = estimate_tokens(text)
    return f"{head_clean}\n\n... ({total_est} tokens total, middle truncated) ...\n\n{tail_clean}"
