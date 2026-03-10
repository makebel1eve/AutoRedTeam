import logging
from typing import List, Optional, Dict, Any, Sequence

from pyrit.executor.attack import AttackExecutor, AttackStrategy, AttackExecutorResult
from pyrit.models import AttackResult, AttackOutcome, Message
from pyrit.memory import CentralMemory, MemoryInterface

from modules.turn import Trajectory, Turn
from modules.embeddings import Embeddings
from modules.db import ThreatDatabase

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        db: ThreatDatabase,
        embedder: Embeddings,
        max_concurrency: int = 5,
    ):
        self.db: ThreatDatabase = db
        self.embedder: Embeddings = embedder
        self.executor: AttackExecutor = AttackExecutor(max_concurrency=max_concurrency)

    async def run_attack(
        self,
        attack: AttackStrategy,
        objectives: List[str],
        memory_labels: Optional[Dict[str, str]] = None,
        **attack_params: Any,
    ) -> List[Trajectory]:
        broadcast = {}
        if memory_labels:
            broadcast["memory_labels"] = memory_labels
        broadcast.update(attack_params)

        result: AttackExecutorResult = await self.executor.execute_attack_async(
            attack=attack,
            objectives=objectives,
            field_overrides=None,
            return_partial_on_failure=True,
            **broadcast,
        )

        memory: MemoryInterface = CentralMemory.get_memory_instance()
        stored_trajectories = []

        for attack_result in result.completed_results:
            messages = memory.get_conversation(
                conversation_id=attack_result.conversation_id
            )
            if not messages:
                logger.warning(
                    f"No messages for conversation {attack_result.conversation_id}"
                )
                continue

            trajectory = self._build_trajectory(messages, attack_result)

            if not trajectory.success:
                logger.info(
                    f"Attack unsuccessful for conversation {attack_result.conversation_id}: "
                    f"{attack_result.outcome_reason}"
                )
                continue

            await self.embedder.generate_embeddings(trajectory)
            await self.db.store_attack_sequence(trajectory)
            logger.info(
                f"Stored trajectory {trajectory.attack_id} ({len(trajectory)} turns)."
            )
            stored_trajectories.append(trajectory)

        for obj, exc in result.incomplete_objectives:
            logger.error(f"Objective '{obj[:60]}' did not complete: {exc}")

        return stored_trajectories

    def _build_trajectory(
        self,
        messages: Sequence[Message],
        attack_result: AttackResult,
    ) -> Trajectory:
        traj = Trajectory()
        traj.attack_id = attack_result.conversation_id
        traj.success = attack_result.outcome == AttackOutcome.SUCCESS

        traj.payload_text = (
            attack_result.last_response.converted_value
            if traj.success and attack_result.last_response
            else ""
        )

        prev_response: Optional[str] = None
        index = 0
        for msg in messages:
            if msg.is_error():
                continue

            role = msg.api_role

            if role == "user":
                turn = Turn(
                    previous_response=prev_response or "",
                    current_prompt=msg.get_value(0),
                    index=index,
                )
                traj.append(turn)
                index += 1
            elif role == "assistant":
                prev_response = msg.get_value(0)

        return traj
