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

1. Install dependencies (including matplotlib for visualization):
   ```bash
   poetry install --no-root
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

3. Place your video as `input.mov` in the project directory

## Usage

### Step 1: Extract frames from video
```bash
# Extract frames at 2 fps (for better data quality and redundancy)
./extract_frames.sh
```

### Step 2: Process frames and analyze

Using the Click CLI tool:

```bash
# Process frames with OpenAI Vision API and analyze results
./uroflow.py read

# Or process with custom output files
./uroflow.py read --output-csv my_data.csv --output-json my_data.json
```

### Step 3: Re-analyze existing data

```bash
# Analyze previously extracted CSV data (includes chart generation)
./uroflow.py analyze

# Analyze without generating chart
./uroflow.py analyze --no-plot

# Or analyze a specific CSV file
./uroflow.py analyze --csv-file my_data.csv
```

### Step 4: Generate visualization chart

```bash
# Create a comprehensive flow chart from existing data
./uroflow.py plot

# Custom output filename
./uroflow.py plot --output my_uroflow_chart.png

# Display chart interactively (requires GUI)
./uroflow.py plot --show
```

### CLI Commands

- `uroflow read` - Processes frame images, extracts weights via OCR, and runs analysis
- `uroflow analyze` - Analyzes existing CSV data and generates chart (use `--no-plot` to skip chart)
- `uroflow plot` - Creates a comprehensive visualization chart with dual-axis flow/volume graph
- `uroflow --help` - Show help and available commands

### Legacy script

The original script is still available:
```bash
python main.py
```

## Output

- `weight_data.csv`: Time-series weight measurements
- `weight_data.json`: Detailed frame-by-frame results
- `uroflow_chart.png`: Comprehensive visualization chart showing:
  - Cumulative volume over time (blue line)
  - Flow rate over time (purple line)
  - Peak flow (Qmax) marker with annotation
  - Average flow rate line
  - Key metrics and clinical interpretation
  - Reference values for comparison

## Note

Since 1g of urine ≈ 1ml, weight changes directly correlate to volume. Flow rate is calculated as the derivative of weight over time.

## Project Roadmap

### Current Features
✅ Frame extraction from video (ffmpeg)
✅ OCR weight reading via OpenAI Vision API
✅ Flow metrics calculation and analysis
✅ Visualization charts with clinical interpretation
✅ CSV/JSON data export

### Planned Enhancements

#### 1. One-Step Processing
- **Goal**: Single command to go from MOV file to complete report
- **Features**:
  - Integrate frame extraction into the Python CLI
  - Auto-detect video format and settings
  - Progress tracking for the entire pipeline

#### 2. PDF Report Generation
- **Goal**: Professional A4 PDF report with complete analysis
- **Features**:
  - Single-page comprehensive report layout
  - Include test metadata (date/time, optional patient identifier)
  - Embed visualization chart
  - Clinical metrics summary
  - Export for medical records

#### 3. Smart Data Management
- **Goal**: Organize test data outside source code with intelligent caching
- **Features**:
  - Structured test sessions (e.g., `~/.uroflow/sessions/YYYY-MM-DD-HHMMSS/`)
  - Idempotent processing (skip already-completed steps)
  - Cache management: reuse frames if video unchanged, reuse CSV if frames unchanged
  - Support multiple tests without data collision
  - Optional test naming/tagging for easy retrieval

#### 4. Standalone macOS Application
- **Goal**: Easy installation without technical prerequisites
- **Features**:
  - Bundle Python, dependencies, and ffmpeg
  - Native macOS app or CLI binary
  - Simple installer (DMG or Homebrew formula)
  - No need for Python/Poetry knowledge
  - Auto-updates for new versions

### Implementation Decisions

#### Technical Choices
- **PDF Generation**: ReportLab for professional medical-grade report control
- **Packaging**: PyInstaller for standalone macOS executable (final step)
- **Data Storage**: Persistent storage in `~/.uroflow/` with session management
- **Frame Extraction**: Keep intermediate frames for potential re-analysis
- **Video Processing**: Integrated ffmpeg subprocess calls

#### Feature Specifications
- **Patient ID**: Optional name field (CLI option or interactive prompt)
- **Data Retention**: Permanent storage of all test sessions
- **Report Format**: Single fixed English template, A4 PDF output
- **Video Support**: MOV format (additional formats as needed)
- **Frame Rate**: 2 fps extraction (configurable if needed)

#### Data Organization Structure
```
~/.uroflow/
├── sessions/
│   ├── 2024-01-15-143022-[patient-name]/
│   │   ├── metadata.json          # Test info, video hash, timestamps
│   │   ├── frames/                # Extracted frame images
│   │   │   ├── frame_0001.jpg
│   │   │   └── ...
│   │   ├── weight_data.csv        # OCR results
│   │   ├── weight_data.json       # Detailed OCR data
│   │   ├── uroflow_chart.png      # Visualization
│   │   └── report.pdf             # Complete analysis report
│   └── latest -> 2024-01-15-143022-[patient-name]/  # Symlink to most recent
└── config.json                    # User preferences

```

#### Processing Pipeline
1. Accept video file from any location (no need to copy to source)
2. Create timestamped session directory
3. Extract frames if not cached (check video hash)
4. Run OCR if CSV doesn't exist
5. Generate analysis and visualization
6. Create PDF report with all results
7. All intermediate files preserved for debugging/re-analysis
