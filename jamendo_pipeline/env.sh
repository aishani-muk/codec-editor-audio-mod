#!/usr/bin/env bash
# Source this to set up env vars for the Celtic Jamendo pipeline.
#
#   source env.sh
#
# Required side effects:
#   - JAMENDO_CLIENT_ID exported (pulled from ~/.jamendo_client_id if present).
#   - JAMENDO_CACHE     exported, defaults to /scratch0/$USER/jamendo.
#   - PYTHONPATH prepended with the parent `modelling/` so we can import
#     siblings like `models.codec_editor` without install.

# ── 1. Jamendo client_id ────────────────────────────────────────────
# The client_id is public; it is safe to commit in plaintext. The
# client_secret is NOT — keep it out of this file and out of git.
#
# Preferred: store just the ID in ~/.jamendo_client_id (one line, chmod 600)
# and let this script pick it up. That way, no secret ever lands in a
# repo file.
if [[ -z "${JAMENDO_CLIENT_ID:-}" ]]; then
    if [[ -r "${HOME}/.jamendo_client_id" ]]; then
        export JAMENDO_CLIENT_ID="$(head -n1 "${HOME}/.jamendo_client_id" | tr -d '[:space:]')"
    else
        # Fallback: the public ID from the plan. If the user rotates their
        # client, they should overwrite ~/.jamendo_client_id to match.
        export JAMENDO_CLIENT_ID="53592e19"
    fi
fi

# ── 2. Scratch cache ────────────────────────────────────────────────
: "${JAMENDO_CACHE:=/scratch0/${USER}/jamendo}"
export JAMENDO_CACHE
mkdir -p "${JAMENDO_CACHE}/mp3" \
         "${JAMENDO_CACHE}/sidecars" \
         "${JAMENDO_CACHE}/tokens" \
         "${JAMENDO_CACHE}/pairs" \
         "${JAMENDO_CACHE}/test_clips" \
         "${JAMENDO_CACHE}/logs"

# ── 3. Python path so we can do `from models.codec_editor import ...` ─
_jp_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
_parent_dir="$( cd -- "${_jp_dir}/.." &> /dev/null && pwd )"
export PYTHONPATH="${_parent_dir}:${_jp_dir}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[jamendo_pipeline env]"
echo "  JAMENDO_CLIENT_ID  = ${JAMENDO_CLIENT_ID:0:4}… (truncated)"
echo "  JAMENDO_CACHE      = ${JAMENDO_CACHE}"
echo "  PYTHONPATH prepend = ${_jp_dir}"
echo "                     + ${_parent_dir}"
