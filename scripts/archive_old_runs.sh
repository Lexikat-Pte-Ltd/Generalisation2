#!/bin/bash

# Check if a folder path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_folder_A>"
    exit 1
fi

# Set the path to folder A from the first command-line argument
folder_A="$1"

# Check if the provided path exists and is a directory
if [ ! -d "$folder_A" ]; then
    echo "Error: '$folder_A' is not a valid directory"
    exit 1
fi

# Get the current date in YYYY_MM_dd format
current_date=$(date +"%Y_%m_%d")

# Count the number of files in folder A
file_count=$(find "$folder_A" -maxdepth 1 -type f | wc -l)

# Create the new subdirectory name
new_dir="archived_${file_count}_${current_date}"

# Create the new subdirectory
mkdir "$folder_A/$new_dir"

# Move all files (not directories) to the new subdirectory
find "$folder_A" -maxdepth 1 -type f -exec mv {} "$folder_A/$new_dir/" \;

echo "Moved $file_count files to $new_dir"