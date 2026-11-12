#!/usr/bin/env bash
set -euo pipefail

root="${1:-${PI05_ROOT:-$HOME/pi05-lora}}"
secret_dir="$root/.secrets"
env_file="$secret_dir/wandb.env"
mkdir -p "$secret_dir" "$root/wandb" "$root/.cache/wandb"
chmod 700 "$secret_dir"

read -r -p "W&B entity (user or team): " entity
read -r -s -p "W&B API key: " api_key
echo
if [[ -z "$entity" || -z "$api_key" ]]; then
  echo "entity and API key are required" >&2
  exit 1
fi

umask 077
{
  printf 'export WANDB_API_KEY=%q\n' "$api_key"
  printf 'export WANDB_ENTITY=%q\n' "$entity"
  printf 'export WANDB_PROJECT=%q\n' "openpi-fr3"
  printf 'export WANDB_DIR=%q\n' "$root/wandb"
  printf 'export WANDB_CACHE_DIR=%q\n' "$root/.cache/wandb"
  printf 'export WANDB_CONFIG_DIR=%q\n' "$root/.config/wandb"
  printf 'export WANDB_DATA_DIR=%q\n' "$root/.local/share/wandb"
} > "$env_file"
chmod 600 "$env_file"
echo "Wrote isolated W&B environment to $env_file (mode 0600)."
echo "No global W&B login or ~/.netrc entry was created."
