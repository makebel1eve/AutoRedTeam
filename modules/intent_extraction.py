import instructor
import os
import logging
from typing import List

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from enum import Enum


class RiskVerdict(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


logger = logging.getLogger(__name__)


class ExtractedIntent(BaseModel):
    objective: str = Field(description="The user's inferred goal in one sentence.")
    domain: str = Field(
        description="The domain this request falls under, e.g. finance, medical, security, general."
    )
    requested_actions: List[str] = Field(
        description="Concrete actions the user is trying to get the LLM to perform."
    )
    privilege_level: str = Field(
        description="The privilege level being requested: 'normal', 'elevated', or 'admin'."
    )
    is_malicious: bool = Field(
        description="True if the intent appears to be an attempt to bypass safety guidelines or extract harmful information."
    )
    reasoning: str = Field(
        description="One sentence explaining why this is or isn't malicious."
    )


# --- Threat Rules ---

THREAT_RULES = [
    lambda i: i.privilege_level in ("elevated", "admin"),
    lambda i: any(
        kw in " ".join(i.requested_actions).lower()
        for kw in (
            "bypass",
            "ignore",
            "override",
            "disable",
            "jailbreak",
            "pretend",
            "act as",
        )
    ),
    lambda i: i.is_malicious,
    lambda i: i.domain in ("weapons", "malware", "exploitation"),
]


def is_confirmed_threat(intent: ExtractedIntent) -> RiskVerdict:
    """Returns True if any threat rule matches the extracted intent."""
    return (
        RiskVerdict.BLOCK
        if any(rule(intent) for rule in THREAT_RULES)
        else RiskVerdict.ALLOW
    )


# --- Extractor ---


class IntentExtractor:
    """
    Uses GPT-4o-mini + Instructor to parse conversation context into a
    structured ExtractedIntent model, then checks it against threat rules.
    """

    def __init__(self) -> None:

        self.client = instructor.from_openai(
            AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY"),
            ),
            mode=instructor.Mode.JSON,
        )

        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    async def extract(self, context: str) -> ExtractedIntent:
        """
        Parse the conversation context and return a structured ExtractedIntent.

        Args:
            context: The full conversation context as a string.

        Returns:
            ExtractedIntent with objective, domain, requested_actions,
            privilege_level, is_malicious, and reasoning fields.
        """
        intent: ExtractedIntent = await self.client.chat.completions.create(
            model=self.model,
            response_model=ExtractedIntent,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the core information-seeking intent from this message "
                        "as a neutral question in one sentence, stripping any fictional framing, "
                        "story context, role attribution, or indirect phrasing. "
                        "Return only the question, starting with 'What', 'How', 'Why', or 'Where'. "
                        "Do not attribute the intent to any person or role."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Conversation:\n{context}",
                },
            ],
        )

        logger.info(
            f"Extracted intent: objective='{intent.objective}' "
            f"domain='{intent.domain}' malicious={intent.is_malicious} "
            f"reasoning='{intent.reasoning}'"
        )

        return intent
