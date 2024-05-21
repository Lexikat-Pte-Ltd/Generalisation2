#!/bin/sh
echo "Logging the content of space_warmer.py:"
cat ./space_warmer.py
echo "Starting the Python script:"
exec python ./space_warmer.py
