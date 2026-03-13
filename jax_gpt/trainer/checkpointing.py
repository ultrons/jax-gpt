from __future__ import annotations

import orbax.checkpoint as ocp
from flax import nnx

from jax_gpt.models.gpt2.config import GPTConfig
from jax_gpt.trainer.config import TrainConfig
from jax_gpt.trainer.train_state import TrainState

# Detect which orbax API is available.
# Modern orbax (>=0.6) exposes ocp.args.StandardSave / StandardRestore.
# Older versions use ocp.PyTreeCheckpointHandler with ocp.AsyncCheckpointer.
_HAS_STANDARD_ARGS = hasattr(ocp, "args") and hasattr(ocp.args, "StandardSave")


class CheckpointManager:
    """
    Async orbax checkpoint manager. Works with local paths and GCS (gs://) paths.

    Saves: step, params (extracted from NNX model), opt_state.
    Does NOT save: model graph structure (reconstructed from GPTConfig at startup).
    """

    def __init__(self, config: TrainConfig, model_config: GPTConfig):
        self._config = config
        self._model_config = model_config

        options = ocp.CheckpointManagerOptions(
            max_to_keep=config.max_checkpoints_to_keep,
            async_options=ocp.AsyncOptions(timeout_secs=300),
        )

        if _HAS_STANDARD_ARGS:
            self._manager = ocp.CheckpointManager(
                directory=config.checkpoint_dir,
                options=options,
            )
        else:
            # Older orbax API: explicit checkpointer mapping.
            checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
            self._manager = ocp.CheckpointManager(
                directory=config.checkpoint_dir,
                checkpointers={"state": checkpointer},
                options=options,
            )

    def save(self, step: int, state: TrainState) -> None:
        """Non-blocking async save. Returns immediately."""
        params = nnx.state(state.model, nnx.Param)
        payload = {
            "step": step,
            "params": params,
            "opt_state": state.opt_state,
        }

        if _HAS_STANDARD_ARGS:
            self._manager.save(
                step,
                args=ocp.args.StandardSave(payload),
            )
        else:
            self._manager.save(step, {"state": payload})

    def restore(self, step: int | None, state: TrainState) -> TrainState:
        """
        Restore from checkpoint.
        If step is None, restores from the latest available checkpoint.
        Returns a TrainState with restored params, opt_state, and step.
        If no checkpoint is found, returns state unchanged.
        """
        restore_step = step if step is not None else self._manager.latest_step()
        if restore_step is None:
            return state

        params_abstract = nnx.state(state.model, nnx.Param)
        abstract_payload = {
            "step": restore_step,
            "params": params_abstract,
            "opt_state": state.opt_state,
        }

        if _HAS_STANDARD_ARGS:
            restored = self._manager.restore(
                restore_step,
                args=ocp.args.StandardRestore(abstract_payload),
            )
        else:
            restored_outer = self._manager.restore(
                restore_step,
                items={"state": abstract_payload},
            )
            restored = restored_outer["state"]

        nnx.update(state.model, restored["params"])

        return TrainState(
            step=restored["step"],
            model=state.model,
            tx=state.tx,
            opt_state=restored["opt_state"],
        )

    def wait_until_finished(self) -> None:
        """Block until any in-flight async save has completed."""
        self._manager.wait_until_finished()

    def latest_step(self) -> int | None:
        """Return the most recently saved step, or None if no checkpoints exist."""
        return self._manager.latest_step()

    def should_save(self, step: int) -> bool:
        """Return True if a checkpoint should be saved at this step."""
        return step % self._config.save_interval == 0
