from pydantic import BaseModel

class ExplainResponse(BaseModel):
  topic: str
  summary: str
  suggestion: str
  sources: list[str]
  tools_used: list[str]