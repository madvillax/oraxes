"""Pydantic schemas for all model-visible tools."""

from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class SearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)


class NavigateArgs(BaseModel):
    url: HttpUrl


class FetchPageArgs(BaseModel):
    url: HttpUrl
    max_chars: int = Field(default=8_000, ge=500, le=20_000)


class ClickArgs(BaseModel):
    selector: str = Field(min_length=1, max_length=500)


class TypeArgs(BaseModel):
    selector: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=3_000)


class ReadPageArgs(BaseModel):
    max_chars: int = Field(default=8_000, ge=500, le=20_000)


class ExtractElementsArgs(BaseModel):
    selector: str = Field(default="body", min_length=1, max_length=500)
    limit: int = Field(default=20, ge=1, le=50)
    max_chars: int = Field(default=8_000, ge=500, le=20_000)


class ScrollArgs(BaseModel):
    pixels: int = Field(default=700, ge=-5_000, le=5_000)


class BackArgs(BaseModel):
    pass


class CollectEvidenceArgs(BaseModel):
    claim: str = Field(min_length=3, max_length=1_000)
    quote: str = Field(min_length=3, max_length=1_500)
    relevance: str = Field(min_length=3, max_length=1_000)


class CompleteResearchArgs(BaseModel):
    reason: str = Field(min_length=3, max_length=500)
