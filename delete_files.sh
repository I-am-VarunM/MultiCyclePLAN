#!/bin/bash

# Define the directories
directory1="/home/varunm/MiniProject/power-side-channel-analysis/plan/pkl"
directory2="/home/varunm/MiniProject/power-side-channel-analysis/plan/modules"

# Remove all files from the first directory
echo "Deleting files from $directory1"
rm -f "${directory1}"/*

# Remove all files from the second directory
echo "Deleting files from $directory2"
rm -f "${directory2}"/*

# Delete specific files from the main directory
echo "Deleting specific files from /home/varunm/MiniProject/power-side-channel-analysis/plan"
rm -f "/home/varunm/MiniProject/power-side-channel-analysis/plan/sigArray_1.pkl" \
    "/home/varunm/MiniProject/power-side-channel-analysis/plan/sigArray_2.pkl" \
    "/home/varunm/MiniProject/power-side-channel-analysis/plan/sigArray_3.pkl"

echo "Deletion completed."

