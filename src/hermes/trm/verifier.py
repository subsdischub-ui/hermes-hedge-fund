"""TRMVerifier — wraps Samsung-SAIL TRM as a structured decision refiner.

We do NOT vendor the TRM source. The Dockerfile clones SamsungSAILMontreal/
TinyRecursiveModels into /opt/trm at build time. If the import fails OR no
trained checkpoint is present on disk, the verifier degrades to a
NullTRMVerifier that passes the LLM action through unchanged with a flat 1.0
confidence (NO random-weight inference on real proposals).

Adaptation: state is encoded as N_STATE_BINS-quantized channel tokens
(`MarketStateEncoder`) in a region of the shared vocab DISJOINT from the
action region; the LLM's draft action is appended as the last 4 cells. The
recursion refines the action cells; state cells are loss-masked at training
time.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from hermes.logging_setup import get_logger
from hermes.settings import get_settings
from hermes.trm.encoder import (
    PAD_ID,
    VOCAB_SIZE,
    ActionCodec,
    MarketStateEncoder,
)

log = get_logger(__name__)

try:
    import torch
    import yaml  # noqa: F401
    _TORCH_OK = True
except ImportError:  # pragma: no cover
    _TORCH_OK = False


class TRMShapeError(ValueError):
    """Raised when the verifier is called with grids of the wrong shape."""


@dataclass
class VerifierOutput:
    refined_action: np.ndarray   # (4,) int32 — refined action tokens
    confidence: float            # sigmoid(q_halt_logits)
    used_trm: bool


class NullTRMVerifier:
    """Pass-through verifier: returns the input action with confidence=1.0.

    Active whenever (a) torch is unavailable, (b) the upstream TRM source
    can't be imported, or (c) no checkpoint is present on disk.
    """

    def __init__(self) -> None:
        self.codec = ActionCodec()
        self.encoder = MarketStateEncoder()

    def refine(
        self,
        state_grid: np.ndarray,
        draft_action: np.ndarray,
        instrument_id: int,
    ) -> VerifierOutput:
        return VerifierOutput(refined_action=draft_action.copy(), confidence=1.0, used_trm=False)


class TRMVerifier:
    """Real TRM verifier — loads checkpoint, runs the recursive forward."""

    def __init__(self, checkpoint: Path, cfg: dict[str, Any]):
        if not _TORCH_OK:
            raise RuntimeError("torch not available")
        if not checkpoint.exists():
            raise RuntimeError(f"TRM checkpoint missing at {checkpoint}")
        self._import_trm()
        self.codec = ActionCodec()
        self.encoder = MarketStateEncoder()
        self.cfg = dict(cfg)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_cfg = dict(self.cfg)
        model_cfg.setdefault("batch_size", 1)
        model_cfg.setdefault("seq_len", cfg["state_len"] + cfg["action_len"])
        model_cfg["vocab_size"] = VOCAB_SIZE
        model_cfg.setdefault("num_puzzle_identifiers", 64)
        model_cfg.setdefault("puzzle_emb_ndim", cfg.get("hidden_size", 256))
        model_cfg.setdefault("puzzle_emb_len", 16)
        model_cfg.setdefault("pos_encodings", "rope")
        model_cfg.setdefault("forward_dtype", "float32")
        model_cfg.setdefault("H_layers", 0)
        model_cfg.setdefault("halt_exploration_prob", 0.1)
        model_cfg.setdefault("no_ACT_continue", True)
        model_cfg.setdefault("mlp_t", False)

        self.model = self._Model(model_cfg).to(self.device).eval()
        sd = torch.load(checkpoint, map_location=self.device, weights_only=False)
        state = sd.get("model", sd)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            log.warning("trm_state_partial", missing=list(missing)[:5], unexpected=list(unexpected)[:5])
        log.info("trm_loaded", checkpoint=str(checkpoint))

    def _import_trm(self) -> None:
        try:
            from models.recursive_reasoning.trm import (  # type: ignore
                TinyRecursiveReasoningModel_ACTV1,
            )
            self._Model = TinyRecursiveReasoningModel_ACTV1
        except ImportError as e:
            raise RuntimeError(
                "TRM source not found. Clone SamsungSAILMontreal/TinyRecursiveModels into "
                "vendor/trm and add it to PYTHONPATH (the Dockerfile does this)."
            ) from e

    @torch.no_grad()
    def refine(
        self, state_grid: np.ndarray, draft_action: np.ndarray, instrument_id: int
    ) -> VerifierOutput:
        if state_grid.shape[0] != self.cfg["state_len"]:
            raise TRMShapeError(
                f"state_grid len {state_grid.shape[0]} != cfg state_len {self.cfg['state_len']}"
            )
        if draft_action.shape[0] != self.cfg["action_len"]:
            raise TRMShapeError(
                f"draft_action len {draft_action.shape[0]} != cfg action_len {self.cfg['action_len']}"
            )
        if not (0 <= instrument_id < self.cfg.get("num_puzzle_identifiers", 64)):
            raise TRMShapeError(f"instrument_id {instrument_id} out of range")

        inputs = np.concatenate([state_grid, draft_action]).astype(np.int32)
        inputs_t = torch.from_numpy(inputs).unsqueeze(0).to(self.device)
        labels_t = torch.full_like(inputs_t, PAD_ID)
        pid_t = torch.tensor([instrument_id], dtype=torch.int32, device=self.device)

        batch = {"inputs": inputs_t, "labels": labels_t, "puzzle_identifiers": pid_t}
        carry = self.model.initial_carry(batch)

        out: dict[str, Any] = {}
        for _ in range(self.cfg.get("halt_max_steps", 6)):
            carry, out = self.model(carry=carry, batch=batch)
            if carry.halted.all().item():
                break

        logits = out["logits"][0]
        action_logits = logits[self.cfg["state_len"]:]
        refined = action_logits.argmax(-1).cpu().numpy().astype(np.int32)
        q_halt = float(torch.sigmoid(out["q_halt_logits"]).item())
        return VerifierOutput(refined_action=refined, confidence=q_halt, used_trm=True)


@lru_cache(maxsize=1)
def get_trm() -> TRMVerifier | NullTRMVerifier:
    s = get_settings()
    if not s.trm_enabled:
        log.info("trm_disabled_by_config")
        return NullTRMVerifier()
    trm_cfg = s.yaml("trm", default={}) or {}
    try:
        return TRMVerifier(checkpoint=s.trm_checkpoint, cfg=trm_cfg)
    except RuntimeError as e:
        log.warning("trm_init_failed_fallback_null", error=str(e))
        return NullTRMVerifier()


def trm_status_summary() -> str:
    """Lightweight, dashboard-facing TRM status string.

    Mirrors get_trm()'s decision path (trm_enabled -> torch available -> checkpoint
    exists on disk) WITHOUT instantiating the model or calling torch.load() a second
    time — a Path.exists() stat() call is cheap, loading checkpoint weights is not.
    Audit finding (docs/SYSTEME_AUTONOME.md partie III, 2026-07-10): TRM is very
    plausibly a NullVerifier in prod (no checkpoint at /data/trm/latest.pt has ever
    been observed) — the downgrade-only contract is correct but inert without a
    trained model. This is surfaced so the operator sees that state, not code that
    silently behaves as if TRM were active.

    NOTE: existence of the checkpoint file does not GUARANTEE TRMVerifier.__init__
    will succeed (state-dict mismatches, a missing upstream TRM import, etc. can
    still fall back to NullTRMVerifier inside get_trm()) — this is an honest
    approximation, not a substitute for actually calling get_trm().
    """
    s = get_settings()
    if not s.trm_enabled:
        return "désactivé (TRM_ENABLED=false)"
    if not _TORCH_OK:
        return "NullVerifier (torch absent — downgrade-only inopérant)"
    if not s.trm_checkpoint.exists():
        return "NullVerifier (checkpoint absent — downgrade-only inopérant ; GEL ratifié 2026-07-11, ré-évaluer l'entraînement à >=100 trades réels clos)"
    return "actif (checkpoint chargé)"
