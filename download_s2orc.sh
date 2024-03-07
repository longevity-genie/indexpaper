#!/bin/bash

# Use s2orc_files.txt as the default file if no argument is provided
FILE=${1:-s2orc_files.txt}
LOGFILE="download_errors.log"

# Clean up previous error log file
rm -f $LOGFILE

# Function to download a file and log errors
download() {
    url=$1
    filename=$(echo $url | awk -F'/' '{print $NF}' | awk -F'?' '{print $1}')
    if ! wget -nc -O "$filename" "$url"; then
        # Log failed download URL to a file
        echo "Failed to download $url" >> $LOGFILE
    fi
}

export -f download

# Read URLs from file line by line and download each in parallel
cat $FILE | xargs -n 1 -P 0 -I {} bash -c 'download "{}"'

# Check if the log file has content and report
if [ -s $LOGFILE ]; then
    echo "Some downloads failed. Check $LOGFILE for details."
else
    echo "All downloads completed successfully."
fi