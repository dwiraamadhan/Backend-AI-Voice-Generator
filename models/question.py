from pydantic import BaseModel
from datetime import datetime


class QuestionClass(BaseModel):
    text: str
    createdAt: datetime
