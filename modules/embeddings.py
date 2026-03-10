import logging
import os
from typing import List

from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from modules.turn import Trajectory

logger = logging.getLogger(__name__)


class Embeddings:
    def __init__(self) -> None:
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self._client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self._groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    async def _normalize_to_intent(self, text: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the core information-seeking intent from this message "
                        "in one neutral sentence, stripping any fictional framing, "
                        "story context, or indirect phrasing. "
                        "Return only the intent sentence, nothing else."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=60,
        )
        if response.choices[0].message.content is None:
            return ""
        normalized = response.choices[0].message.content.strip()
        logger.info(f"Normalized: '{text[:60]}' → '{normalized}'")
        return normalized

    async def generate_embeddings(self, trajectory: Trajectory) -> None:
        for turn in trajectory.trajectory:
            if not turn.combined_prompt:
                turn.combined_prompt = (
                    f"Assistant: {turn.previous_response}\nUser: {turn.current_prompt}"
                    if turn.previous_response
                    else f"User: {turn.current_prompt}"
                )

            if turn.combined_prompt_embedding is None:
                turn.combined_prompt_embedding = await self.embed_text(
                    turn.combined_prompt
                )

            if not turn.intent:
                turn.intent = await self._normalize_to_intent(turn.current_prompt)

            if turn.intent_embedding is None:
                turn.intent_embedding = await self.embed_text(turn.intent)

        if trajectory.payload_text and trajectory.payload_embedding is None:
            trajectory.payload_embedding = await self.embed_text(
                trajectory.payload_text
            )

        logger.info(f"Generated embeddings for trajectory {trajectory.attack_id}")

    async def embed_text(self, text: str) -> List[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    async def normalize_and_embed(self, text: str) -> tuple[str, List[float]]:
        normalized = await self._normalize_to_intent(text)
        embedding = await self.embed_text(normalized)
        return normalized, embedding
