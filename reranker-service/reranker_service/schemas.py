from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_DOCUMENTS_PER_REQUEST = 20


class RerankDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @field_validator("id", "text")
    @classmethod
    def require_nonblank_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value must not be blank.")
        return value


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    documents: list[RerankDocument] = Field(max_length=MAX_DOCUMENTS_PER_REQUEST)
    top_n: int | None = Field(default=None, ge=1, le=MAX_DOCUMENTS_PER_REQUEST)

    @field_validator("query")
    @classmethod
    def require_nonblank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must not be blank.")
        return value


class RerankResult(BaseModel):
    id: str
    score: float
    rank: int = Field(ge=1)


class RerankResponse(BaseModel):
    results: list[RerankResult]
