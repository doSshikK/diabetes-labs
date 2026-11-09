from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FeatureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(ge=-1.0, le=1.0)
    sex: float = Field(ge=-1.0, le=1.0)
    bmi: float = Field(ge=-1.0, le=1.0)
    bp: float = Field(ge=-1.0, le=1.0)
    s1: float = Field(ge=-1.0, le=1.0)
    s2: float = Field(ge=-1.0, le=1.0)
    s3: float = Field(ge=-1.0, le=1.0)
    s4: float = Field(ge=-1.0, le=1.0)
    s5: float = Field(ge=-1.0, le=1.0)
    s6: float = Field(ge=-1.0, le=1.0)

    def values_for(self, features: list[str]) -> list[float]:
        payload = self.model_dump()
        return [payload[name] for name in features]


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    model_sha256: str


class PromoteRequest(BaseModel):
    version: str = Field(min_length=1)
