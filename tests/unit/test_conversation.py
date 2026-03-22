"""Tests for ConversationStore."""

import pytest
from pathlib import Path
from graph.conversation import ConversationStore


@pytest.fixture
def store(tmp_path):
    return ConversationStore(data_dir=tmp_path)


def test_create_and_load(store):
    conv = store.create("sess_1", "user_a")
    assert conv["session_id"] == "sess_1"
    assert conv["user_id"] == "user_a"
    assert conv["turns"] == []

    loaded = store.load("sess_1")
    assert loaded is not None
    assert loaded["session_id"] == "sess_1"


def test_load_nonexistent(store):
    assert store.load("no_such_session") is None


def test_append_turn(store):
    store.create("sess_2", "user_b")
    store.append_turn("sess_2", "user", "hello")
    store.append_turn("sess_2", "assistant", "hi there")

    conv = store.load("sess_2")
    assert len(conv["turns"]) == 2
    assert conv["turns"][0]["role"] == "user"
    assert conv["turns"][1]["content"] == "hi there"


def test_append_turn_auto_creates(store):
    """Appending to a nonexistent session should auto-create it."""
    store.append_turn("sess_auto", "user", "hello")
    conv = store.load("sess_auto")
    assert conv is not None
    assert len(conv["turns"]) == 1


def test_turn_limit(store):
    """Should cap at 50 turns."""
    store.create("sess_cap", "user_c")
    for i in range(60):
        store.append_turn("sess_cap", "user", f"msg {i}")
    conv = store.load("sess_cap")
    assert len(conv["turns"]) == 50
    assert conv["turns"][0]["content"] == "msg 10"


def test_save_directives(store):
    store.create("sess_dir", "user_d")
    store.save_directives("sess_dir", ["删除前要确认", "只处理广告邮件"])
    store.save_directives("sess_dir", ["删除前要确认", "新约束"])

    directives = store.get_directives("sess_dir")
    assert len(directives) == 3
    assert "删除前要确认" in directives
    assert "新约束" in directives


def test_get_history_for_prompt(store):
    store.create("sess_hist", "user_e")
    store.append_turn("sess_hist", "user", "what is 1+1?")
    store.append_turn("sess_hist", "assistant", "2")
    store.append_turn("sess_hist", "user", "and 2+2?")

    history = store.get_history_for_prompt("sess_hist")
    assert "User: what is 1+1?" in history
    assert "Assistant: 2" in history
    assert "User: and 2+2?" in history


def test_get_history_empty_session(store):
    assert store.get_history_for_prompt("nonexistent") == ""


def test_list_sessions(store):
    store.create("sess_x1", "user_f")
    store.append_turn("sess_x1", "user", "first conversation")
    store.create("sess_x2", "user_f")
    store.append_turn("sess_x2", "user", "second conversation")
    store.create("sess_x3", "user_g")  # different user
    store.append_turn("sess_x3", "user", "other user")

    sessions = store.list_sessions("user_f")
    assert len(sessions) == 2
    session_ids = {s["session_id"] for s in sessions}
    assert "sess_x1" in session_ids
    assert "sess_x2" in session_ids


def test_list_sessions_no_empty(store):
    """Sessions with no turns should not appear in list."""
    store.create("sess_empty", "user_h")
    sessions = store.list_sessions("user_h")
    assert len(sessions) == 0
