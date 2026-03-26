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
    """Should keep turns bounded via auto-archive + hard cap."""
    store.create("sess_cap", "user_c")
    for i in range(60):
        store.append_turn("sess_cap", "user", f"msg {i}")
    conv = store.load("sess_cap")
    # Auto-archiving may reduce turns below 50; hard cap still at 50
    assert len(conv["turns"]) <= 50
    # Archived topics should exist if turns were compressed
    archive = conv.get("archive", {})
    total = len(conv["turns"]) + sum(
        len(t.get("turns", [])) for t in archive.get("topics", [])
    )
    assert total >= 50  # no turns lost, just archived


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


# ── Context CRUD (/modify support) ──────────────────────────

def test_list_turns(store):
    store.create("sess_lt", "u")
    store.append_turn("sess_lt", "user", "hello")
    store.append_turn("sess_lt", "assistant", "hi")
    turns = store.list_turns("sess_lt")
    assert len(turns) == 2
    assert turns[0]["role"] == "user"


def test_delete_turn(store):
    store.create("sess_dt", "u")
    store.append_turn("sess_dt", "user", "msg0")
    store.append_turn("sess_dt", "assistant", "msg1")
    store.append_turn("sess_dt", "user", "msg2")
    assert store.delete_turn("sess_dt", 1)
    turns = store.list_turns("sess_dt")
    assert len(turns) == 2
    assert turns[1]["content"] == "msg2"


def test_delete_turn_out_of_range(store):
    store.create("sess_dtor", "u")
    store.append_turn("sess_dtor", "user", "msg0")
    assert not store.delete_turn("sess_dtor", 5)
    assert not store.delete_turn("sess_dtor", -1)


def test_delete_turns_range(store):
    store.create("sess_dr", "u")
    for i in range(6):
        store.append_turn("sess_dr", "user", f"msg{i}")
    count = store.delete_turns_range("sess_dr", 1, 4)
    assert count == 3
    turns = store.list_turns("sess_dr")
    assert len(turns) == 3
    assert turns[0]["content"] == "msg0"
    assert turns[1]["content"] == "msg4"


def test_edit_turn(store):
    store.create("sess_et", "u")
    store.append_turn("sess_et", "user", "old content")
    assert store.edit_turn("sess_et", 0, "new content")
    turns = store.list_turns("sess_et")
    assert turns[0]["content"] == "new content"


def test_edit_turn_out_of_range(store):
    store.create("sess_etor", "u")
    assert not store.edit_turn("sess_etor", 0, "x")


def test_clear_turns(store):
    store.create("sess_ct", "u")
    store.append_turn("sess_ct", "user", "hello")
    store.append_turn("sess_ct", "assistant", "hi")
    assert store.clear_turns("sess_ct")
    turns = store.list_turns("sess_ct")
    assert len(turns) == 0


def test_get_summary_empty(store):
    store.create("sess_gs", "u")
    assert store.get_summary("sess_gs") == ""


# ── Persistent directives (cross-session) ────────────────────

def test_persistent_directives_empty(store):
    assert store.load_persistent_directives("user_pd") == []


def test_add_persistent_directive(store):
    assert store.add_persistent_directive("user_pd", "never delete without asking")
    assert store.add_persistent_directive("user_pd", "use Chinese")
    directives = store.load_persistent_directives("user_pd")
    assert len(directives) == 2
    assert "never delete without asking" in directives


def test_add_persistent_directive_dedup(store):
    store.add_persistent_directive("user_dup", "rule1")
    assert not store.add_persistent_directive("user_dup", "rule1")
    assert len(store.load_persistent_directives("user_dup")) == 1


def test_remove_persistent_directive(store):
    store.add_persistent_directive("user_rm", "a")
    store.add_persistent_directive("user_rm", "b")
    store.add_persistent_directive("user_rm", "c")
    assert store.remove_persistent_directive("user_rm", 1)
    directives = store.load_persistent_directives("user_rm")
    assert directives == ["a", "c"]


def test_remove_persistent_directive_invalid(store):
    assert not store.remove_persistent_directive("user_no", 0)


def test_get_all_directives_merged(store):
    # Persistent
    store.add_persistent_directive("user_merged", "persistent rule")
    # Session
    store.create("sess_merged", "user_merged")
    store.save_directives("sess_merged", ["session rule"])
    # Merge
    all_dirs = store.get_all_directives("sess_merged", user_id="user_merged")
    assert len(all_dirs) == 2
    assert all_dirs[0] == "persistent rule"  # persistent first
    assert all_dirs[1] == "session rule"


def test_get_all_directives_dedup(store):
    store.add_persistent_directive("user_dd", "same rule")
    store.create("sess_dd", "user_dd")
    store.save_directives("sess_dd", ["same rule"])
    all_dirs = store.get_all_directives("sess_dd", user_id="user_dd")
    assert len(all_dirs) == 1
