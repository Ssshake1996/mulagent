"""Tests for dispatcher node."""

import pytest
from graph.dispatcher import dispatch_node, _dispatch_by_keywords


def test_keyword_code():
    result = _dispatch_by_keywords("write a Python function to sort a list")
    assert result["intent"] == "code"


def test_keyword_research():
    result = _dispatch_by_keywords("search for the latest AI news")
    assert result["intent"] == "research"


def test_keyword_data():
    result = _dispatch_by_keywords("analyze this CSV data and make a chart")
    assert result["intent"] == "data"


def test_keyword_writing():
    result = _dispatch_by_keywords("write an article about climate change")
    assert result["intent"] == "writing"


def test_keyword_reasoning():
    result = _dispatch_by_keywords("calculate the compound interest on $1000")
    assert result["intent"] == "reasoning"


def test_keyword_general():
    result = _dispatch_by_keywords("hello how are you")
    assert result["intent"] == "general"


def test_keyword_chinese_code():
    result = _dispatch_by_keywords("帮我写一个排序函数")
    assert result["intent"] == "code"


def test_keyword_chinese_research():
    result = _dispatch_by_keywords("搜索最新的AI新闻")
    assert result["intent"] == "research"


def test_complexity_simple():
    result = _dispatch_by_keywords("write hello world")
    assert result["complexity"] == "simple"


def test_complexity_complex():
    long_input = "write a function that " + "does something complex " * 10
    result = _dispatch_by_keywords(long_input)
    assert result["complexity"] == "complex"


@pytest.mark.asyncio
async def test_dispatch_node_no_llm():
    state = {"user_input": "debug this Python code"}
    result = await dispatch_node(state, llm=None)
    assert result["intent"] == "code"
    assert "status" in result
