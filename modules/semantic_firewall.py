from typing import Dict, List, Optional
import math
import logging

from modules.intent_extraction import IntentExtractor, is_confirmed_threat, RiskVerdict
from semantic_kernel.functions import kernel_function
from modules.db import ThreatDatabase
from modules.embeddings import Embeddings


logger = logging.getLogger(__name__)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    text-embedding-3 outputs are L2-normalised, so this reduces to a dot product,
    but we keep the full formula for correctness with any embedding model.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class FirewallPlugin:
    """
    Semantic Kernel plugin that implements a real-time firewall using EWMA-CUSUM
    to detect multi-turn attack escalation.

    Risk is scored as a weighted combination of three signals:

        R = w1 * S_traj + w2 * S_pos + w3 * S_payload

    where:
        S_traj    – cosine similarity between the current interaction pair and
                    the most similar stored attack turn.
        S_pos     – positional weight; turns that are late in a known attack
                    chain are scored higher (monotonically increasing with
                    turn_index / total_turns).
        S_payload – cosine similarity between the current user message and the
                    final malicious payload of the matched attack.
    """

    def __init__(
        self,
        db: ThreatDatabase,
        embedder: Embeddings,
        ewma_lambda: float = 0.8,
        cusum_threshold: float = 1.0,
        target_mean: float = 0.0,
        slack: float = 0.1,
        w1: float = 0.5,
        w2: float = 0.3,
        w3: float = 0.2,
        context: str = "",
    ):
        """
        Args:
            db:               ThreatDatabase instance for similarity queries.
            embedder:         Embeddings instance for generating query embeddings.
            ewma_lambda:      Smoothing factor for EWMA (0 < lambda <= 1).
            cusum_threshold:  CUSUM upper control limit; block when exceeded.
            target_mean:      Target mean of the risk signal (0 = no attack).
            slack:            Allowance for natural variability (reference value k).
            w1:               Weight for trajectory similarity (S_traj).
            w2:               Weight for positional signal (S_pos).
            w3:               Weight for payload similarity (S_payload).
        """
        self.db = db
        self.embedder = embedder
        self.ewma_lambda = ewma_lambda
        self.cusum_threshold = cusum_threshold
        self.target_mean = target_mean
        self.slack = slack
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.context = context
        self.intent_extractor = IntentExtractor()

        self._state: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _compute_s_pos(turn_index: int, total_turns: int) -> float:
        """
        Positional score S_pos: a monotonically increasing value in [0, 1].

        A turn late in an attack chain (e.g. turn 8 of 10) carries more risk
        than an early probe (turn 1 of 10) because the conversation has already
        undergone significant escalation.

        We use a power curve so scores grow slowly at first and
        accelerate toward the end of the sequence:

            S_pos = (turn_index / total_turns) ** 1.5

        The exponent > 1 de-emphasises early turns and emphasises late ones.
        """
        if total_turns <= 0:
            return 0.0
        relative_pos = turn_index / total_turns
        return relative_pos**1.5

    @staticmethod
    def _compute_s_payload(
        user_message_emb: List[float],
        payload_embedding: Optional[List[float]],
    ) -> float:
        """
        Payload similarity S_payload: cosine similarity between the current
        user message and the final malicious payload of the matched attack.

        A high score means the user's current message closely resembles what
        the attacker was ultimately trying to extract — a strong signal even
        if the attack has only just begun.
        """
        if not payload_embedding:
            return 0.0
        return max(0.0, _cosine_similarity(user_message_emb, payload_embedding))

    def _compute_risk(
        self,
        best_turn: Dict,
        user_message_emb: List[float],
    ) -> float:
        """
        Combine the three signals into a single risk score R.

        Args:
            best_turn:        The highest-similarity row returned by the DB.
            user_message_emb: Embedding of the current user message (not the pair).

        Returns:
            R in [0, 1] (approximately; can slightly exceed 1 if weights sum > 1).
        """
        raw_sim = float(best_turn["similarity"])
        s_traj = raw_sim**0.5  # sqrt to amplify raw_sim

        s_pos = self._compute_s_pos(
            turn_index=best_turn["turn_index"],
            total_turns=best_turn["total_turns"],
        )

        s_payload = self._compute_s_payload(
            user_message_emb=user_message_emb,
            payload_embedding=best_turn.get("payload_embedding"),
        )

        R = self.w1 * s_traj + self.w2 * s_pos + self.w3 * s_payload

        logger.info(
            f"Risk breakdown — S_traj={s_traj:.3f}, S_pos={s_pos:.3f}, "
            f"S_payload={s_payload:.3f} → R={R:.3f} "
            f"(turn {best_turn['turn_index']}/{best_turn['total_turns']})"
        )
        return R

    @kernel_function(
        description="Analyzes a user message for potential attack patterns using EWMA-CUSUM."
    )
    async def analyze_risk(
        self,
        user_message: str,
        conversation_id: str,
        last_response: str,
    ) -> RiskVerdict:
        """
        Compute risk score and update EWMA-CUSUM statistics.

        Returns:
            RiskVerdict.ALLOW  – risk is low; pass message to the LLM.
            RiskVerdict.BLOCK  – risk is high; block and log.
        """
        normalized, intent_emb = await self.embedder.normalize_and_embed(user_message)
        combined = (
            f"Assistant: {last_response}\nUser: {user_message}"
            if last_response
            else f"User: {user_message}"
        )

        user_msg_emb: List[float] = await self.embedder.embed_text(user_message)

        self.context += "\n" + combined

        similar_turns = await self.db.find_similar_turns(
            query_embedding=intent_emb,
            top_k=5,
            min_similarity=0.2,
        )

        state = self._state.setdefault(
            conversation_id,
            {"ewma": 0.0, "cusum_high": 0.0, "cusum_low": 0.0},
        )
        if similar_turns:
            best_turn = similar_turns[0]
            R = self._compute_risk(best_turn, user_msg_emb)
        else:
            R = 0.0

        logger.info(f"Risk score R={R:.3f} for conv {conversation_id}")

        ewma_new = self.ewma_lambda * R + (1 - self.ewma_lambda) * state["ewma"]
        state["ewma"] = ewma_new

        cusum_high_new = max(
            0.0,
            state["cusum_high"] + (ewma_new - self.target_mean) - self.slack,
        )
        state["cusum_high"] = cusum_high_new

        cusum_low_new = max(
            0.0,
            state["cusum_low"] - (ewma_new - self.target_mean) - self.slack,
        )
        state["cusum_low"] = cusum_low_new

        # 7. Decision logic.
        if cusum_high_new > self.cusum_threshold:
            logger.info(
                f"BLOCK: CUSUM={cusum_high_new:.2f} > {self.cusum_threshold} "
                f"(conv {conversation_id})"
            )
            return RiskVerdict.BLOCK

        if cusum_high_new >= 1.0:
            logger.debug(f"Intent extraction triggered for conv {conversation_id}")
            return await self._extract_intent(self.context)
        return RiskVerdict.ALLOW

    async def _extract_intent(self, context: str) -> RiskVerdict:
        """Structured intent extraction for medium-risk messages."""
        intent = await self.intent_extractor.extract(context)
        return is_confirmed_threat(intent)
