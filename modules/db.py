import asyncpg
from pgvector.asyncpg import register_vector
from modules.turn import Trajectory
from modules.embeddings import Embeddings
from typing import List, Dict, Optional
import os
import logging

logger = logging.getLogger(__name__)


class ThreatDatabase:
    def __init__(self, embedder: Embeddings):
        self.embedder = embedder
        self.conn: Optional[asyncpg.Connection]

    async def init_pool(self) -> None:
        """Open a single connection and ensure table exists."""
        self.conn = await asyncpg.connect(
            host=os.getenv("POSTGRES_HOST"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            port=int(os.getenv("POSTGRES_PORT", 5432)),
        )
        await register_vector(self.conn)
        await self.create_table_if_not_exists()

    async def create_table_if_not_exists(self) -> None:
        if self.conn is None:
            return

        await self.conn.execute("""
            CREATE SCHEMA IF NOT EXISTS threat_intel;

            CREATE TABLE IF NOT EXISTS threat_intel.attack_turns (
                id SERIAL PRIMARY KEY,
                attack_id UUID NOT NULL,
                turn_index INT NOT NULL,
                total_turns INT NOT NULL,
                user_prompt TEXT NOT NULL,
                previous_response TEXT,
                intent_text TEXT,
                intent_embedding vector(1024),
                payload_text TEXT,
                user_prompt_embedding vector(1024),
                previous_response_embedding vector(1024),
                combined_prompt_embedding vector(1024),
                payload_embedding vector(1024),
                success BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_attack_turns_combined_embedding
                ON threat_intel.attack_turns
                USING ivfflat (combined_prompt_embedding vector_cosine_ops)
                WITH (lists = 100);
        """)

    async def store_attack_sequence(self, trajectory: Trajectory) -> None:
        if not trajectory.success:
            return

        for turn in trajectory.trajectory:
            if turn.combined_prompt_embedding is None:
                raise ValueError(
                    f"Turn {turn.index} missing combined_prompt_embedding. Call generate_embeddings() first."
                )
            if self.conn is None:
                return
            await self.conn.execute(
                """
                INSERT INTO threat_intel.attack_turns (
                    attack_id, turn_index, total_turns, user_prompt,
                    previous_response, payload_text, user_prompt_embedding,
                    previous_response_embedding, combined_prompt_embedding,
                    payload_embedding, intent_text, intent_embedding, success
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                trajectory.attack_id,
                turn.index,
                len(trajectory),
                turn.current_prompt,
                turn.previous_response,
                trajectory.payload_text,
                turn.user_prompt_embedding,
                turn.previous_response_embedding,
                turn.combined_prompt_embedding,
                trajectory.payload_embedding,
                turn.intent,
                turn.intent_embedding,
                trajectory.success,
            )

    async def find_similar_turns(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.6,
    ) -> List[Dict]:
        if self.conn is None:
            return []

        rows = await self.conn.fetch(
            """
            SELECT
                attack_id, turn_index, total_turns, user_prompt,
                previous_response, payload_text, payload_embedding,
                1 - (intent_embedding <-> $1) AS similarity
            FROM threat_intel.attack_turns
            WHERE 1 - (intent_embedding <-> $1) >= $2
            ORDER BY intent_embedding <-> $1
            LIMIT $3
            """,
            query_embedding,
            min_similarity,
            top_k,
        )

        results = []
        for row in rows:
            raw_payload_emb = row["payload_embedding"]
            results.append(
                {
                    "attack_id": row["attack_id"],
                    "turn_index": row["turn_index"],
                    "total_turns": row["total_turns"],
                    "user_prompt": row["user_prompt"],
                    "previous_response": row["previous_response"],
                    "payload_text": row["payload_text"],
                    "payload_embedding": list(raw_payload_emb)
                    if raw_payload_emb is not None
                    else None,
                    "similarity": float(row["similarity"]),
                }
            )
        return results

    async def close(self) -> None:
        if self.conn is not None:
            await self.conn.close()
