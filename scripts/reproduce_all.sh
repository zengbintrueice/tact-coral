#!/usr/bin/env bash
# Reproduce everything: main + supplementary.
set -e
bash scripts/reproduce_main.sh
bash scripts/reproduce_supplementary.sh
echo "ALL DONE"
