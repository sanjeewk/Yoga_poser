from typing import Literal, Optional
from pydantic import BaseModel, Field


class PoseCatalogEntry(BaseModel):
    key: str
    display_name: str
    sanskrit: str
    description: str


POSE_CATALOG: list[PoseCatalogEntry] = [
    PoseCatalogEntry(key="tadasana", display_name="Mountain", sanskrit="Tadasana",
                     description="Neutral standing baseline; feet together, arms at sides."),
    PoseCatalogEntry(key="adho_mukha_svanasana", display_name="Downward-Facing Dog",
                     sanskrit="Adho Mukha Svanasana",
                     description="Inverted V; hands and feet on floor, hips lifted."),
    PoseCatalogEntry(key="virabhadrasana_i", display_name="Warrior I",
                     sanskrit="Virabhadrasana I",
                     description="Deep lunge, back foot angled, arms raised overhead."),
    PoseCatalogEntry(key="virabhadrasana_ii", display_name="Warrior II",
                     sanskrit="Virabhadrasana II",
                     description="Deep lunge, arms extended horizontally, gaze forward."),
    PoseCatalogEntry(key="vrksasana", display_name="Tree", sanskrit="Vrksasana",
                     description="Standing balance; one foot on inner thigh, hands in prayer."),
    PoseCatalogEntry(key="bhujangasana", display_name="Cobra", sanskrit="Bhujangasana",
                     description="Prone backbend; chest lifted, arms supporting."),
    PoseCatalogEntry(key="balasana", display_name="Child's Pose", sanskrit="Balasana",
                     description="Kneeling fold; hips to heels, arms forward or beside."),
    PoseCatalogEntry(key="marjaryasana", display_name="Cat", sanskrit="Marjaryasana",
                     description="Tabletop with spine rounded upward."),
]


class FeedbackHint(BaseModel):
    joint: str
    cue: str
    severity: Literal["minor", "major"]


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    landmarks: Optional[list[list[float]]] = None
    feedback: list[FeedbackHint] = Field(default_factory=list)
    hold_seconds: float = 0.0
    rep_count: int = 0


class PredictionRequest(BaseModel):
    session_id: str
    landmarks: Optional[list[tuple[float, float, float, float]]] = Field(
        default=None, min_length=33, max_length=33,
    )


class SessionStartRequest(BaseModel):
    target_poses: Optional[list[str]] = None


class SessionStartResponse(BaseModel):
    session_id: str
    target_poses: list[str]


class SessionStatusResponse(BaseModel):
    current_pose: Optional[str]
    hold_seconds: float
    rep_count_per_pose: dict[str, int]
    history: list[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    landmark_runtime: Literal["browser"]

    model_config = {"protected_namespaces": ()}
