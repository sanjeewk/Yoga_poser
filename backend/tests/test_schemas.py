from backend.app.schemas import (
    POSE_CATALOG, PredictionResponse, FeedbackHint, SessionStartRequest,
)


def test_catalog_has_8_poses():
    keys = {p.key for p in POSE_CATALOG}
    assert keys == {
        "tadasana", "adho_mukha_svanasana", "virabhadrasana_i",
        "virabhadrasana_ii", "vrksasana", "bhujangasana",
        "balasana", "marjaryasana",
    }


def test_prediction_response_optional_fields():
    r = PredictionResponse(label="tadasana", confidence=0.9, landmarks=None,
                           feedback=[], hold_seconds=0.0, rep_count=0)
    assert r.label == "tadasana"
    assert r.feedback == []


def test_feedback_hint_severity_validates():
    h = FeedbackHint(joint="left_knee", cue="Straighten your front knee", severity="major")
    assert h.severity == "major"


def test_session_start_request_default():
    r = SessionStartRequest()
    assert r.target_poses is None
