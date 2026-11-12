#!/usr/bin/env python3
"""Register the success-only, strict-quality and hiveboard (with and without state) FR3 Pi configs."""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "\nif len({config.name for config in _CONFIGS}) != len(_CONFIGS):"
BLOCK_START = "\n\n# FR3 Pi 0.5 dataset ablations."
BLOCK = """

# FR3 Pi 0.5 dataset ablations.  They inherit the LoRA/model/optimizer settings of
# the baseline; the hiveboard runs additionally use OpenPI's default LoRA batch of 32
# with the default cosine schedule peak, and keep only the latest ~12.5 GB checkpoint
# (still saved every save_interval steps for resuming).
_FR3_BASELINE = next(c for c in _CONFIGS if c.name == "pi05_fr3_lora_success_and_suboptimal")
_FR3_HIVEBOARD = dict(
    batch_size=32,
    lr_schedule=dataclasses.replace(_FR3_BASELINE.lr_schedule, peak_lr=2.5e-5, decay_lr=2.5e-6),
    keep_period=None,
)
_CONFIGS.extend(
    [
        dataclasses.replace(
            _FR3_BASELINE,
            name="pi05_fr3_lora_success_only",
            data=dataclasses.replace(
                _FR3_BASELINE.data,
                repo_id="local/fr3_hiveboard_success_only_train",
            ),
        ),
        dataclasses.replace(
            _FR3_BASELINE,
            name="pi05_fr3_lora_strict_quality",
            data=dataclasses.replace(
                _FR3_BASELINE.data,
                repo_id="local/fr3_hiveboard_strict_quality_train",
            ),
        ),
        dataclasses.replace(
            _FR3_BASELINE,
            name="pi05_fr3_lora_hiveboard",
            data=dataclasses.replace(
                _FR3_BASELINE.data,
                repo_id="local/fr3_hiveboard_train",
            ),
            **_FR3_HIVEBOARD,
        ),
        # Same data with the state discretized into the prompt, as pi05_droid and
        # pi05_aloha do; the baseline above keeps pi05_libero's state-free input.
        dataclasses.replace(
            _FR3_BASELINE,
            name="pi05_fr3_lora_hiveboard_state",
            model=dataclasses.replace(_FR3_BASELINE.model, discrete_state_input=True),
            data=dataclasses.replace(
                _FR3_BASELINE.data,
                repo_id="local/fr3_hiveboard_train",
            ),
            freeze_filter=dataclasses.replace(
                _FR3_BASELINE.model, discrete_state_input=True
            ).get_freeze_filter(),
            **_FR3_HIVEBOARD,
        ),
        dataclasses.replace(
            _FR3_BASELINE,
            name="pi05_fr3_lora_datahive_state",
            model=dataclasses.replace(_FR3_BASELINE.model, discrete_state_input=True),
            data=dataclasses.replace(
                _FR3_BASELINE.data,
                repo_id="local/fr3_datahive_train",
            ),
            freeze_filter=dataclasses.replace(
                _FR3_BASELINE.model, discrete_state_input=True
            ).get_freeze_filter(),
            **_FR3_HIVEBOARD,
        ),
    ]
)
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    text = args.config.read_text()
    if BLOCK in text:
        print("Pi variant configs already installed")
        return
    if MARKER not in text:
        raise RuntimeError("could not find _CONFIGS uniqueness marker")
    if BLOCK_START in text:
        start, end = text.index(BLOCK_START), text.index(MARKER)
        if start > end:
            raise RuntimeError("installed Pi variant block is not directly before the marker")
        args.config.write_text(text[:start] + BLOCK + text[end:])
        print(f"updated Pi variant configs in {args.config}")
        return
    args.config.write_text(text.replace(MARKER, BLOCK + MARKER, 1))
    print(f"installed Pi variant configs in {args.config}")


if __name__ == "__main__":
    main()
