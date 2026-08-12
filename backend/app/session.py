import uuid
from collections import deque

from backend.app.schemas import SessionStatusResponse

HOLD_MIN_CONFIDENCE = 0.6
HOLD_STABILITY_WINDOW = 1.0
REP_MIN_HOLD_SECONDS = 3.0
REP_EXIT_DURATION = 0.5
REP_COOLDOWN = 0.5


class SessionState:
    def __init__(self, session_id: str, target_poses: list[str]):
        self.session_id = session_id
        self.target_poses = target_poses
        self._current_pose: str | None = None
        self._current_since_ts: float = 0.0
        self._hold_seconds: float = 0.0
        self._last_update_ts: float = 0.0
        self._label_history: deque[tuple[float, str]] = deque(maxlen=60)
        self._rep_count_per_pose: dict[str, int] = {}
        self._history: list[dict] = []
        self._rep_cooldown_until: float = 0.0

    def update(self, label: str, confidence: float, has_major_feedback: bool, now_ts: float):
        prev_ts = self._last_update_ts
        self._last_update_ts = now_ts
        self._label_history.append((now_ts, label))

        if label != self._current_pose:
            self._on_pose_exit(now_ts)
            self._current_pose = label
            self._current_since_ts = now_ts
            self._hold_seconds = 0.0
            return

        stable = self._is_stable(label, now_ts)
        if (stable and confidence >= HOLD_MIN_CONFIDENCE and not has_major_feedback):
            self._hold_seconds += max(0.0, now_ts - prev_ts)
        self._current_pose = label

    def _is_stable(self, label: str, now_ts: float) -> bool:
        window_start = now_ts - HOLD_STABILITY_WINDOW
        recent = [(t, l) for t, l in self._label_history if t >= window_start]
        return bool(recent) and all(l == label for _, l in recent)

    def _on_pose_exit(self, now_ts: float):
        if self._current_pose is None or self._current_pose == "unknown":
            return
        if self._hold_seconds >= REP_MIN_HOLD_SECONDS and now_ts >= self._rep_cooldown_until:
            self._rep_count_per_pose[self._current_pose] = (
                self._rep_count_per_pose.get(self._current_pose, 0) + 1
            )
            self._history.append({
                "pose": self._current_pose,
                "held_seconds": round(self._hold_seconds, 2),
                "ended_at": now_ts,
            })
            self._rep_cooldown_until = now_ts + REP_COOLDOWN

    def snapshot(self) -> SessionStatusResponse:
        return SessionStatusResponse(
            current_pose=self._current_pose,
            hold_seconds=round(self._hold_seconds, 2),
            rep_count_per_pose=dict(self._rep_count_per_pose),
            history=list(self._history),
        )

    def reset(self):
        self._current_pose = None
        self._current_since_ts = 0.0
        self._hold_seconds = 0.0
        self._label_history.clear()
        self._rep_count_per_pose.clear()
        self._history.clear()
        self._rep_cooldown_until = 0.0


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, SessionState] = {}
        self._last_active: dict[str, float] = {}

    def start(self, target_poses: list[str]) -> SessionState:
        sid = uuid.uuid4().hex
        s = SessionState(sid, target_poses)
        self._sessions[sid] = s
        self._last_active[sid] = 0.0
        return s

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def evict_idle(self, now_ts: float, max_age_seconds: float = 1800):
        stale = [sid for sid, t in self._last_active.items() if now_ts - t > max_age_seconds]
        for sid in stale:
            self._sessions.pop(sid, None)
            self._last_active.pop(sid, None)
