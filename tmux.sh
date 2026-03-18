#!/bin/bash

source ./.venv/bin/activate

tmux new-session -d -s RagEngine

tmux rename-window -t RagEngine:0 'nvim'

tmux send-keys -t RagEngine:1 'nvim' C-m

tmux new-window -t RagEngine:2 

tmux attach -t RagEngine


