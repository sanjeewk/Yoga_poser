from backend.app.session import SessionState, SessionStore


def test_hold_accrues_when_stable_and_confident():
    s = SessionState("s1", ["tadasana"])
    for t in range(0, 30):
        s.update("tadasana", 0.9, False, float(t) / 10)
    snap = s.snapshot()
    assert snap.hold_seconds >= 1.5
    assert snap.current_pose == "tadasana"


def test_hold_does_not_accrue_with_major_feedback():
    s = SessionState("s1", ["tadasana"])
    for t in range(0, 30):
        s.update("tadasana", 0.9, True, float(t) / 10)
    snap = s.snapshot()
    assert snap.hold_seconds == 0.0


def test_rep_counted_after_three_second_hold_then_exit():
    s = SessionState("s1", ["tadasana"])
    for t in range(0, 40):
        s.update("tadasana", 0.9, False, float(t) / 10)
    for t in range(40, 50):
        s.update("unknown", 0.2, False, float(t) / 10)
    snap = s.snapshot()
    assert snap.rep_count_per_pose.get("tadasana", 0) == 1


def test_store_start_and_get():
    store = SessionStore()
    s = store.start(["tadasana"])
    assert store.get(s.session_id) is s


def test_store_evicts_idle():
    store = SessionStore()
    s = store.start(["tadasana"])
    store.evict_idle(now_ts=10000.0, max_age_seconds=10)
    assert store.get(s.session_id) is None
