"""Tests for gateway.adapter — SessionManager and ProgressEvent."""

import pytest
from pathlib import Path

from gateway.adapter import ProgressEvent, SessionManager
from graph.conversation import ConversationStore


@pytest.fixture
def mgr(tmp_path):
    store = ConversationStore(data_dir=tmp_path)
    return SessionManager(store)


# ── ProgressEvent ────────────────────────────────────────────────

def test_progress_event_defaults():
    e = ProgressEvent()
    assert e.round_num == 0
    assert e.action == ""
    assert e.detail == ""
    assert e.elapsed_ms == 0


def test_progress_event_fields():
    e = ProgressEvent(round_num=3, action="tool_call", detail="web_search", elapsed_ms=120)
    assert e.round_num == 3
    assert e.action == "tool_call"


# ── SessionManager.get_or_create ─────────────────────────────────

def test_get_or_create_returns_stable_id(mgr):
    sid1 = mgr.get_or_create("user1", "cli")
    sid2 = mgr.get_or_create("user1", "cli")
    assert sid1 == sid2


def test_get_or_create_different_context(mgr):
    sid_cli = mgr.get_or_create("user1", "cli")
    sid_feishu = mgr.get_or_create("user1", "chat_abc")
    assert sid_cli != sid_feishu


# ── SessionManager.new_session ───────────────────────────────────

def test_new_session_creates_conversation(mgr):
    sid = mgr.new_session("user1", "cli")
    conv = mgr.conv_store.load(sid)
    assert conv is not None
    assert conv["user_id"] == "user1"


def test_new_session_replaces_current(mgr):
    sid1 = mgr.get_or_create("user1", "cli")
    sid2 = mgr.new_session("user1", "cli")
    assert sid1 != sid2
    # After new_session, get_or_create returns the new one
    assert mgr.get_or_create("user1", "cli") == sid2


# ── SessionManager.resume_session ────────────────────────────────

def test_resume_existing_session(mgr):
    sid = mgr.new_session("user1", "cli")
    mgr.conv_store.append_turn(sid, "user", "hello")
    # Create a different session
    mgr.new_session("user1", "cli")
    # Resume the original
    ok = mgr.resume_session("user1", "cli", sid)
    assert ok is True
    assert mgr.get_or_create("user1", "cli") == sid


def test_resume_nonexistent_session(mgr):
    ok = mgr.resume_session("user1", "cli", "no_such_session")
    assert ok is False


# ── SessionManager.list_sessions ─────────────────────────────────

def test_list_sessions(mgr):
    sid = mgr.new_session("user1", "cli")
    mgr.conv_store.append_turn(sid, "user", "hello")
    sessions = mgr.list_sessions("user1")
    assert len(sessions) >= 1
    sids = [s["session_id"] for s in sessions]
    assert sid in sids


def test_list_sessions_empty(mgr):
    sessions = mgr.list_sessions("nobody")
    assert sessions == []


# ── SessionManager.ensure_conversation ───────────────────────────

def test_ensure_conversation_creates_if_missing(mgr):
    mgr.ensure_conversation("brand_new_sess", "user1")
    conv = mgr.conv_store.load("brand_new_sess")
    assert conv is not None


def test_ensure_conversation_no_overwrite(mgr):
    sid = mgr.new_session("user1", "cli")
    mgr.conv_store.append_turn(sid, "user", "keep this")
    mgr.ensure_conversation(sid, "user1")
    conv = mgr.conv_store.load(sid)
    assert len(conv["turns"]) == 1
    assert conv["turns"][0]["content"] == "keep this"


# ── SessionManager.conv_store ────────────────────────────────────

def test_conv_store_accessor(mgr):
    assert isinstance(mgr.conv_store, ConversationStore)
