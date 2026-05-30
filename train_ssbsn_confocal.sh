#!/bin/bash
set -e

GPU="0,1"
CONFIG=SSBSN_CONFOCAL
THREAD=8

echo "=== Train SS-BSN CONFOCAL ==="
mkdir -p ./dataset
python train.py -c $CONFIG -g $GPU --thread $THREAD

echo "=== Done ==="
