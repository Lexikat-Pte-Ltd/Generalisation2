#!/bin/bash
SESSION_NAME="main-script-session"
PYTHON_SCRIPT="./scripts/main.py"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Create new tmux session and run Python script
tmux new-session -d -s "$SESSION_NAME" "python3 $PYTHON_SCRIPT"