#!/bin/bash

# Extract frames at 2 fps for better data quality and redundancy
# This gives us 2 samples per second to detect and filter out bad readings
ffmpeg -i "$1" -vf fps=2 -q:v 2 frame_%04d.jpg
