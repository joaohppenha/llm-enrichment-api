from enum import Enum
from pydantic import BaseModel, Field

class CategoryEnum(str, Enum):
    BILLING = "billing"
    BUG = "bug"
    FEATURE = "feature"
    LEGAL = "legal"
    OTHER = "other"

class UrgencyEnum(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

class NoticeInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Notice or support message text")

class EnrichmentOutput(BaseModel):
    category: CategoryEnum
    urgency: UrgencyEnum
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str = Field(..., description="One sentence summary of the input")
    reason: str = Field(..., description="One short sentence explaining the classification")