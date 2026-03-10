import os
from enum import Enum

from pyrit.executor.attack import CrescendoAttack, PromptSendingAttack
from pyrit.executor.attack import AttackAdversarialConfig
from pyrit.prompt_target import OpenAIChatTarget
import logging

logger = logging.getLogger("__main__")


def _make_azure_target() -> OpenAIChatTarget:
    logger.info("Made Target")
    return OpenAIChatTarget(
        endpoint="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("GROQ_MODEL"),
    )


def _make_azure_target_small() -> OpenAIChatTarget:
    return OpenAIChatTarget(
        endpoint="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("GROQ_MODEL"),
    )


class AttackType(str, Enum):
    CRESCENDO = "crescendo"
    SINGLE_TURN = "single_turn"


def build_attack(
    attack_type: AttackType,
    max_turns: int = 10,
    max_backtracks: int = 10,
):
    """
    Build a configured attack strategy pointed at the staging clone
    (same Azure deployment as production — swap endpoint env var to isolate).

    Args:
        attack_type:    CRESCENDO or SINGLE_TURN.
        max_turns:      Crescendo only — max conversation turns.
        max_backtracks: Crescendo only — max backtracks before giving up.

    Returns:
        A ready-to-run AttackStrategy instance.
    """
    objective_target = _make_azure_target()

    if attack_type == AttackType.CRESCENDO:
        print(
            f"Building Crescendo with max_turns = {max_turns} and max_backtracks = {max_backtracks}"
        )
        adversarial_target = _make_azure_target_small()
        return CrescendoAttack(
            objective_target=objective_target,
            attack_adversarial_config=AttackAdversarialConfig(
                target=adversarial_target
            ),
            max_turns=max_turns,
            max_backtracks=max_backtracks,
        )

    if attack_type == AttackType.SINGLE_TURN:
        print("building single turn")
        return PromptSendingAttack(
            objective_target=objective_target,
        )

    raise ValueError(f"Unknown attack type: {attack_type}")
