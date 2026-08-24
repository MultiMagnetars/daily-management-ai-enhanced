from pydantic import BaseModel, Field, field_validator
import re

class Structure(BaseModel):
    abstract_translation: str = Field(
        description="complete, sentence-by-sentence professional Chinese translation of the supplied abstract"
    )
    tldr: str = Field(
        description="one or two sentences stating the management research question or object and the most important explicit finding"
    )
    motivation: str = Field(
        description="research question, theory or research background, and literature gap supported by title and abstract"
    )
    method: str = Field(
        description="explicit data, sample, variables or research object, design, and method; adapt to non-empirical paper types"
    )
    result: str = Field(
        description="explicit main conclusions, mechanisms, heterogeneity, and economic consequences with evidence strength preserved"
    )
    conclusion: str = Field(
        description="explicit theoretical or literature contribution and supported practical or policy implications"
    )
