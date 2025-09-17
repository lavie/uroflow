# DIY Uroflowmetry Analysis Tool

A simple system for extracting uroflow measurements from video recordings of a digital scale, providing an accessible alternative to specialized medical equipment.

## Overview

This project enables uroflowmetry testing using commonly available equipment:
- A digital scale to measure collected urine weight
- A camera/phone to record the scale display
- OpenAI's Vision API to extract weight readings from video frames

Uroflowmetry measures urine flow rate over time to assess bladder and urethral function, helping diagnose conditions like BPH, strictures, and bladder outlet obstruction.

## Workflow

1. **Record**: Video the digital scale display during urination (save as `input.mov`)
2. **Extract frames**: Run `./extract_frames.sh` to extract 1 frame per second
3. **Process**: Run `python main.py` to extract weight readings via OCR
4. **Analyze**: Use the generated CSV/JSON data to calculate flow parameters

## Key Measurements

From the weight-over-time data, you can derive:
- **Peak flow rate (Qmax)**: Maximum ml/s achieved
- **Average flow rate (Qave)**: Mean flow throughout voiding  
- **Voided volume**: Total amount in ml
- **Flow curve shape**: Diagnostic patterns

## Setup

1. Install dependencies:
   ```bash
   poetry install --no-root
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

3. Place your video as `input.mov` in the project directory

## Usage

```bash
# Extract frames from video (1 fps)
./extract_frames.sh

# Process frames to extract weight readings
python main.py
```

## Output

- `weight_data.csv`: Time-series weight measurements
- `weight_data.json`: Detailed frame-by-frame results

## Note

Since 1g of urine ≈ 1ml, weight changes directly correlate to volume. Flow rate is calculated as the derivative of weight over time.
