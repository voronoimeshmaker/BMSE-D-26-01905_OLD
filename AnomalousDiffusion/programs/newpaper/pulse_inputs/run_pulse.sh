#!/bin/bash
# Run all TSOM Gaussian-pulse benchmark cases
# Usage: bash run_pulse.sh [path_to_executable]

EXE="${1:-./tsom_pulse}"

echo "Running P1a..."
$EXE P1a.yaml && echo "P1a done" || echo "P1a FAILED"
echo ""

echo "Running P1b..."
$EXE P1b.yaml && echo "P1b done" || echo "P1b FAILED"
echo ""

echo "Running P1c..."
$EXE P1c.yaml && echo "P1c done" || echo "P1c FAILED"
echo ""

echo "Running P2a..."
$EXE P2a.yaml && echo "P2a done" || echo "P2a FAILED"
echo ""

echo "Running P2b..."
$EXE P2b.yaml && echo "P2b done" || echo "P2b FAILED"
echo ""

echo "Running P2c..."
$EXE P2c.yaml && echo "P2c done" || echo "P2c FAILED"
echo ""

echo "Running P3a..."
$EXE P3a.yaml && echo "P3a done" || echo "P3a FAILED"
echo ""

echo "Running P3b..."
$EXE P3b.yaml && echo "P3b done" || echo "P3b FAILED"
echo ""

echo "Running P3c..."
$EXE P3c.yaml && echo "P3c done" || echo "P3c FAILED"
echo ""

echo "All cases finished."
