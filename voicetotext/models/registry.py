"""Declarations of the ML model artifacts the app downloads."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    approx_bytes: int
    kind: str  # "hf_snapshot"


# Sizes from docs/IMPLEMENTATION_SPEC.md (verified against release/HF APIs).
PARAKEET = ModelSpec(
    key="parakeet",
    repo_id="csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    approx_bytes=640 * 1024 * 1024,
    kind="hf_snapshot",
)
# NOTE: "csukuangfj/sherpa-onnx-silero-vad" (the placeholder from the plan)
# does not exist on Hugging Face (verified: 404). "csukuangfj/vad" is the
# real repo containing silero_vad.onnx / silero_vad_v5.onnx; substituted
# per the implementer note. Re-verify before Task 6 wires up real downloads.
SILERO_VAD = ModelSpec(
    key="silero_vad",
    repo_id="csukuangfj/vad",
    approx_bytes=2 * 1024 * 1024,
    kind="hf_snapshot",
)
NLLB = ModelSpec(
    key="nllb",
    repo_id="JustFrederik/nllb-200-distilled-600M-ct2-int8",
    approx_bytes=700 * 1024 * 1024,
    kind="hf_snapshot",
)

ALL: dict[str, ModelSpec] = {m.key: m for m in (PARAKEET, SILERO_VAD, NLLB)}
