from pydantic import BaseModel

class ExplainRequest(BaseModel):
  text: str

class AgentInitRequest(BaseModel):
  model: str

class EmbeddingRequest(BaseModel):
  model: str
  text: str