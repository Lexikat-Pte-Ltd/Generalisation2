#!/bin/bash

set_random_permissions() {
    local directory=$1
    local file
    for file in "$directory"/*; do
        if [ -d "$file" ]; then
            set_random_permissions "$file"
        elif [ -f "$file" ]; then
            chmod "$(printf '%o' $((RANDOM%512)))" "$file"
        fi
    done
}

set_random_permissions "$(pwd)"