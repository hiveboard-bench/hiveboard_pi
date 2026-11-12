SHELL := /bin/bash
.DEFAULT_GOAL := help
.NOTPARALLEL:

# Configuration ---------------------------------------------------------------
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
-include $(ROOT)/local.mk
PI05_ROOT ?= $(ROOT)
SOURCE ?= datahive
DATAHIVE_ROOT ?= $(HOME)/DataHive
DATAHIVE_PATH := $(abspath $(DATAHIVE_ROOT))
ifeq ($(SOURCE),datahive)
INPUT_DIR ?= $(DATAHIVE_ROOT)/samples
MANIFEST ?= $(PI05_ROOT)/manifests/datahive.jsonl
TRAIN_CONFIG := pi05_fr3_lora_datahive_state
EXP_NAME ?= datahive-state
else ifneq ($(filter oopsie reference,$(SOURCE)),)
INPUT_DIR ?= $(ROOT)/data/dataset_hiveboard
MANIFEST ?= $(ROOT)/training/openpi/experiments/manifests/hiveboard/success_and_suboptimal.jsonl
TRAIN_CONFIG := pi05_fr3_lora_hiveboard_state
EXP_NAME ?= hiveboard-state-repro
else
$(error SOURCE must be datahive, oopsie or reference)
endif
FRONT_CAMERA ?= front
WRIST_CAMERA ?= wrist
STATE_GRIPPER_UNIT ?= width_m
GRIPPER_OPEN_WIDTH ?=
BINARY_OPEN_VALUE ?=
TASK_MAP ?=
STEPS ?= 10
XLA_PYTHON_CLIENT_MEM_FRACTION ?= 0.80
export PI05_ROOT INPUT_DIR EXP_NAME STEPS XLA_PYTHON_CLIENT_MEM_FRACTION
export SOURCE MANIFEST TRAIN_CONFIG FRONT_CAMERA WRIST_CAMERA STATE_GRIPPER_UNIT GRIPPER_OPEN_WIDTH BINARY_OPEN_VALUE TASK_MAP

.PHONY: help setup wandb manifest prepare verify check stats smoke train resume evaluate test

help: ## Show available commands without starting training.
	@printf '\nHiveboard pi0.5 — source: %s\n\n' "$(SOURCE)"
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  make %-10s %s\n", $$1, $$2}' $(ROOT)/Makefile
	@printf '\nData: %s\nExperiment: %s\n\n' "$(INPUT_DIR)" "$(EXP_NAME)"

# Installation and data -------------------------------------------------------
setup: ## Install OpenPI with Python 3.11 and the frozen uv.lock.
	@bash "$(ROOT)/scripts/hiveboard.sh" setup

wandb: ## Configure W&B (prompts for entity and a hidden API key).
	@bash "$(ROOT)/training/openpi/configure_wandb.sh" "$(PI05_ROOT)"

manifest: ## Select successful DataHive episodes and create splits (seed 42).
	@bash "$(ROOT)/scripts/hiveboard.sh" manifest

prepare: manifest ## Convert to LeRobot; set INPUT_DIR=/path/to/samples.
	@bash "$(ROOT)/scripts/hiveboard.sh" prepare

# Validation ------------------------------------------------------------------
verify: ## Check source integrity and model configuration.
	@bash "$(ROOT)/scripts/hiveboard.sh" verify

check: ## Also check converted episode and frame counts.
	@bash "$(ROOT)/scripts/hiveboard.sh" check

stats: check ## Compute normalization for the selected dataset.
	@bash "$(ROOT)/scripts/hiveboard.sh" stats

smoke: stats ## Run a separate short training job (STEPS=10, range 1–50).
	@bash "$(ROOT)/scripts/hiveboard.sh" smoke

# Training --------------------------------------------------------------------
train: stats ## Start training in a new experiment and W&B run.
	@bash "$(ROOT)/training/openpi/run_training.sh" "$(EXP_NAME)" "$(TRAIN_CONFIG)"

resume: check ## Resume EXP_NAME after checking data and normalization.
	@bash "$(ROOT)/training/openpi/run_training.sh" "$(EXP_NAME)" "$(TRAIN_CONFIG)" --resume

evaluate: check ## Evaluate checkpoint 29999 on validation/test; save local JSON.
	@bash "$(ROOT)/scripts/hiveboard.sh" evaluate

test: ## Run DataHive and original-format integration tests.
	@test -f "$(DATAHIVE_PATH)/src/datahive/episode.py" || { echo "DataHive checkout not found. Set DATAHIVE_ROOT=/path/to/DataHive." >&2; exit 1; }
	@cd "$(PI05_ROOT)/openpi" && PYTHONPATH="$(DATAHIVE_PATH)/src$${PYTHONPATH:+:$$PYTHONPATH}" uv run --frozen python -m pytest "$(ROOT)/tests" -q
