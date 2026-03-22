"""Tests for token estimation and truncation."""

from common.tokenizer import estimate_tokens, truncate_to_tokens, truncate_middle


def test_empty_text():
    assert estimate_tokens("") == 0


def test_english_text():
    text = "Hello world this is a test"
    tokens = estimate_tokens(text)
    # 6 words * 1.3 ≈ 8 tokens, plus spaces/punct
    assert 5 < tokens < 20


def test_chinese_text():
    text = "你好世界这是一个测试"
    tokens = estimate_tokens(text)
    # 9 CJK chars * 1.5 ≈ 14 tokens
    assert 10 < tokens < 25


def test_mixed_text():
    text = "Hello 你好 world 世界"
    tokens = estimate_tokens(text)
    assert tokens > 5


def test_code_text():
    text = "def hello():\n    print('hello world')\n    return 42"
    tokens = estimate_tokens(text)
    assert tokens > 10


def test_truncate_short_text():
    """Short text should not be truncated."""
    text = "short text"
    result = truncate_to_tokens(text, 100)
    assert result == text


def test_truncate_long_text():
    """Long text should be truncated."""
    text = "\n".join(f"line {i}: hello world content" for i in range(200))
    result = truncate_to_tokens(text, 50)
    assert len(result) < len(text)
    assert "truncated" in result


def test_truncate_tail():
    """Tail truncation keeps the end."""
    lines = [f"line {i}" for i in range(100)]
    text = "\n".join(lines)
    result = truncate_to_tokens(text, 30, tail=True)
    assert "line 99" in result


def test_truncate_middle():
    """Middle truncation keeps head and tail."""
    lines = [f"line {i}: some content here" for i in range(200)]
    text = "\n".join(lines)
    result = truncate_middle(text, 100)
    assert "line 0" in result
    assert "line 199" in result
    assert "middle truncated" in result


def test_estimate_tokens_numbers():
    text = "12345 67890"
    tokens = estimate_tokens(text)
    assert tokens > 0


def test_estimate_tokens_json():
    """JSON-like content should get reasonable estimates."""
    text = '{"name": "test", "value": 42, "items": [1, 2, 3]}'
    tokens = estimate_tokens(text)
    assert 15 < tokens < 60
