#!/usr/bin/env bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

filename=$1

awk '{print $12}' "$filename" | sort | uniq -c | awk '{print $2 ": " $1}'
