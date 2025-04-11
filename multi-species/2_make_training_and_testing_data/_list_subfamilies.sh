#!/usr/bin/env bash

# Check if a file name was provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <filename> [search_term]"
    exit 1
fi

filename=$1
search_term=${2:-SINE}  # Default to "LINE" if not provided

# Check if the file exists
if [ ! -f "$filename" ]; then
    echo "File not found: $filename"
    exit 1
fi

# Process the file
echo "SINES"
awk -F'\t' -v term="$search_term" '
    $12 == term {
        count[$11]++
    }
    END {
        for (item in count) {
            print item ": " count[item]
        }
    }
' "$filename" | sort -t: -k2 -nr
