#!/bin/bash
# Doppio clic dal Finder: apre il visualizzatore del G1.
cd "$(dirname "$0")"
exec ./.venv/bin/mjpython vis_g1.py
