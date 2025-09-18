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

### Quick Start: One-Step Processing

```bash
# Process a video file from start to finish
./uroflow.py process video.mov

# You'll be prompted for patient name (optional)
# The system will:
# 1. Create a timestamped session in ~/.uroflow/sessions/
# 2. Extract frames from video (if not already cached)
# 3. Process frames with OCR (if not already done)
# 4. Generate analysis and visualization chart
```

### Session Management

All data is organized in sessions under `~/.uroflow/sessions/`:

```bash
# List all sessions
./uroflow.py sessions

# Work with latest session
./uroflow.py analyze  # Re-analyze latest session
./uroflow.py plot     # Regenerate chart for latest

# Work with specific session
./uroflow.py analyze --session 2024-01-15-143022
./uroflow.py plot --session 2024-01-15-143022-John_Doe
```

### Individual Commands

#### Process Video (Recommended)
```bash
./uroflow.py process input.mov --patient-name "John Doe"
./uroflow.py process input.mov --fps 3  # Custom frame rate
./uroflow.py process input.mov --force  # Force re-processing, ignore all cached data
```

#### Manual Step-by-Step (if needed)
```bash
# 1. Extract frames only
./extract_frames.sh input.mov  # Legacy bash script

# 2. Run OCR on frames
./uroflow.py read --session latest
./uroflow.py read --force  # Force re-run OCR, delete cached results

# 3. Analyze data
./uroflow.py analyze --session latest

# 4. Generate chart
./uroflow.py plot --session latest
```

### CLI Commands

- `uroflow process <video>` - Complete pipeline: video → frames → OCR → analysis → chart
- `uroflow sessions` - List all analysis sessions with their status
- `uroflow read` - Process frame images with OCR (uses session management)
- `uroflow analyze` - Analyze data and generate chart (default: latest session)
- `uroflow plot` - Create visualization chart (default: latest session)
- `uroflow --help` - Show all available commands

## Output

- `weight_data.csv`: Time-series weight measurements with frame numbers
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
✅ CSV data export
✅ Session-based data management in `~/.uroflow/sessions/`
✅ Smart caching (skip completed steps)
✅ One-command processing from video to analysis

### Planned Enhancements

#### 1. One-Step Processing ✅ COMPLETED
- **Implemented**: `uroflow process <video>` command
- **Features**:
  - ✅ Integrated frame extraction via ffmpeg subprocess
  - ✅ Smart caching based on video hash
  - ✅ Progress tracking with status messages
  - ✅ Session management with patient names

#### 2. PDF Report Generation
- **Goal**: Professional A4 PDF report with complete analysis
- **Features**:
  - Single-page comprehensive report layout
  - Include test metadata (date/time, optional patient identifier)
  - Embed visualization chart
  - Clinical metrics summary
  - Export for medical records

#### 3. Smart Data Management ✅ COMPLETED
- **Implemented**: Filesystem-based session management
- **Features**:
  - ✅ Structured sessions: `~/.uroflow/sessions/YYYY-MM-DD-HHMMSS-[patient]/`
  - ✅ Idempotent processing (checks filesystem for completed steps)
  - ✅ Smart caching: video hash validation, skip existing frames/OCR
  - ✅ Multiple tests without collision
  - ✅ Patient name tagging in session directories
  - ✅ Filesystem as single source of truth (no redundant metadata)

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

#### Data Organization Structure (Implemented)
```
~/.uroflow/
└── sessions/
    ├── 2024-01-15-143022-John_Doe/
    │   ├── metadata.json          # Minimal: patient name, video hash
    │   ├── frames/                # Extracted frame images
    │   │   ├── frame_0001.jpg
    │   │   └── ...
    │   ├── weight_data.csv        # OCR results
    │   ├── uroflow_chart.png      # Visualization
    │   └── report.pdf             # (Coming in Phase 2)
    └── latest -> 2024-01-15-143022-John_Doe/  # Symlink to most recent
```

**Session Status Detection (filesystem-based):**
- Frames extracted: `frames/*.jpg` files exist
- OCR completed: `weight_data.csv` exists
- Analysis done: `uroflow_chart.png` exists
- Report generated: `report.pdf` exists (Phase 2)

#### Processing Pipeline (Implemented)
1. ✅ Accept video file from any location
2. ✅ Create timestamped session directory with patient name
3. ✅ Extract frames if not cached (validates video hash)
4. ✅ Run OCR if CSV doesn't exist
5. ✅ Generate analysis and visualization chart
6. ⏳ Create PDF report (Phase 2 - pending)
7. ✅ All intermediate files preserved for re-analysis

### Next Steps

**Performance Optimization: Concurrent OCR Processing**
- Add concurrent/parallel processing for OCR API calls
- Implement exponential backoff and retry logic for API rate limits
- Refactor frame validation to occur after all OCR completes (since frames won't be processed in order)
- Expected speedup: 5-10x for OCR phase
- Consider using `asyncio` or `concurrent.futures` for parallel API calls

**Phase 2: PDF Report Generation** (Ready to implement)
- Add ReportLab for professional PDF generation
- Single-page A4 report with chart, metrics, and clinical interpretation
- Include test date/time and patient name

**Phase 4: macOS Packaging** (After core features complete)
- PyInstaller for standalone executable
- Bundle all dependencies including ffmpeg
