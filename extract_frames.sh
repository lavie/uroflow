#!/bin/bash


ffmpeg -i input.mov -vf fps=1 -q:v 2 frame_%04d.jpg
