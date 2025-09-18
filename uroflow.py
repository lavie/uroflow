#!/usr/bin/env python3
import os
import base64
import json
import glob
from openai import OpenAI
import time
import csv
import click
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
from session_manager import SessionManager
import subprocess
import shutil
import asyncio
from async_ocr import AsyncOCRProcessor, get_max_concurrent, get_max_per_second
from pdf_report import UroflowReportGenerator

# Constants for validation and analysis
MAX_FLOW_RATE = 20.0  # ml/s - maximum physiologically realistic flow rate
MIN_WEIGHT_CHANGE_THRESHOLD = 0.5  # grams - minimum change to detect start of urination
MIN_FLOW_THRESHOLD = 0.1  # ml/s - minimum flow to consider as active voiding
API_DELAY = 0.5  # seconds - delay between API calls to avoid rate limiting
FPS = 2  # frames per second extracted from video
FRAME_INTERVAL = 1.0 / FPS  # time interval between frames (0.5 seconds)
MAX_INTRA_SECOND_DIFF = 2.0  # max acceptable weight difference within same second

# Clinical reference values (adult male)
NORMAL_QMAX_MIN = 15.0  # ml/s - minimum normal peak flow
NORMAL_QAVE_MIN = 10.0  # ml/s - minimum normal average flow
NORMAL_VOLUME_MIN = 150.0  # ml - minimum normal voided volume
NORMAL_VOLUME_MAX = 500.0  # ml - maximum normal voided volume
BORDERLINE_QMAX = 10.0  # ml/s - borderline peak flow threshold

# Smoothing parameters
# Configure smoothing in seconds for clarity (converted to points based on 2 fps sampling)
SMOOTHING_SECONDS = float(os.getenv('UROFLOW_SMOOTHING_SECONDS', '8.0'))  # Default 8 seconds
SMOOTHING_WINDOW_SIZE = int(SMOOTHING_SECONDS * 2)  # Convert to points at 2 fps
MIN_SUSTAINED_DURATION = float(os.getenv('UROFLOW_MIN_SUSTAINED_DURATION', '2.0'))  # 2-second rule for Qmax

# OpenAI API configuration
VISION_MODEL = "gpt-4o"
MAX_TOKENS = 100

# Set up the client using environment variable
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def smooth_flow_rates(flow_rates, window_size=None):
    """
    Apply moving average smoothing to flow rate data

    Args:
        flow_rates: List of flow rate values
        window_size: Size of the smoothing window (default: SMOOTHING_WINDOW_SIZE)

    Returns:
        List of smoothed flow rate values
    """
    if window_size is None:
        window_size = SMOOTHING_WINDOW_SIZE

    if len(flow_rates) <= 1 or window_size <= 1:
        return flow_rates

    smoothed = []
    for i in range(len(flow_rates)):
        # Calculate window bounds
        half_window = window_size // 2
        start = max(0, i - half_window)
        end = min(len(flow_rates), i + half_window + 1)

        # Calculate moving average
        window_values = flow_rates[start:end]
        smoothed.append(sum(window_values) / len(window_values))

    return smoothed


def calculate_qmax_2sec(flow_times, flow_rates, min_duration=None, already_smoothed=False):
    """
    Calculate Qmax using the 2-second rule:
    Find the highest flow rate sustained for at least min_duration seconds

    Args:
        flow_times: List of time points
        flow_rates: List of flow rate values (can be raw or already smoothed)
        min_duration: Minimum sustained duration in seconds (default: MIN_SUSTAINED_DURATION)
        already_smoothed: If True, skip smoothing (data is already smoothed)

    Returns:
        Tuple of (qmax_value, qmax_index) where index is in the smoothed data
    """
    if min_duration is None:
        min_duration = MIN_SUSTAINED_DURATION

    if not flow_rates:
        return 0, 0

    if len(flow_rates) < 2:
        return max(flow_rates), 0

    # Only smooth if not already smoothed
    if already_smoothed:
        smoothed_rates = flow_rates
    else:
        smoothed_rates = smooth_flow_rates(flow_rates)

    # Find the highest sustained peak
    max_sustained = 0
    max_sustained_index = 0

    for i in range(len(smoothed_rates)):
        current_rate = smoothed_rates[i]

        # Skip if this rate is lower than what we've already found
        if current_rate <= max_sustained:
            continue

        start_time = flow_times[i]

        # Check how long this flow rate is sustained (within 10% tolerance)
        threshold = current_rate * 0.9
        sustained_until = i

        for j in range(i + 1, len(smoothed_rates)):
            if smoothed_rates[j] >= threshold:
                sustained_until = j
            else:
                break

        # Calculate duration this rate was sustained
        if sustained_until > i:
            sustained_duration = flow_times[sustained_until] - start_time
            if sustained_duration >= min_duration:
                # Find the actual peak within this sustained period
                # Use the middle of the sustained period or where the actual max occurs
                period_values = smoothed_rates[i:sustained_until+1]
                local_max = max(period_values)
                local_max_offset = period_values.index(local_max)
                max_sustained = local_max
                max_sustained_index = i + local_max_offset

    # If no sustained peak found (e.g., very short test), use smoothed max
    if max_sustained == 0:
        max_val = max(smoothed_rates)
        max_sustained_index = smoothed_rates.index(max_val)
        return max_val, max_sustained_index

    return max_sustained, max_sustained_index

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_weight_from_image(image_path):
    """Send image to OpenAI Vision API and extract weight reading"""
    try:
        # Encode the image
        base64_image = encode_image(image_path)
        
        # Create the message
        response = client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Look at this image of a digital scale display. Extract ONLY the numerical weight reading shown on the display. Return just the number with its decimal point if present, nothing else. If you can't clearly see a number, return 'unclear'."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        
        weight = response.choices[0].message.content.strip()
        return weight
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "error"

def validate_and_aggregate_results(raw_results, discarded_reasons):
    """Validate and aggregate results after all OCR is complete"""
    # Sort by frame number since async results arrive out of order
    raw_results.sort(key=lambda x: x['frame'])

    # Validate sequence
    validated_results = []
    last_valid_weight = 0.0
    last_valid_time = -1
    discarded_count = 0

    for result in raw_results:
        # Skip error results in validation
        if result.get('error'):
            validated_results.append(result)
            continue

        try:
            weight_value = float(result['weight'])

            # Check for non-decreasing constraint
            if weight_value < last_valid_weight:
                click.echo(f"\n  ⚠️  Frame {result['frame']}: Weight decreased from {last_valid_weight} to {weight_value} - discarding")
                discarded_count += 1
                discarded_reasons['decreasing'] += 1
                continue

            # Check for unrealistic jumps (>20g/s flow rate)
            if last_valid_time >= 0:
                time_diff = result['time_seconds'] - last_valid_time
                if time_diff > 0:
                    weight_diff = weight_value - last_valid_weight
                    flow_rate = weight_diff / time_diff
                    if flow_rate > MAX_FLOW_RATE:
                        click.echo(f"\n  ⚠️  Frame {result['frame']}: Unrealistic flow rate of {flow_rate:.1f} ml/s - discarding")
                        discarded_count += 1
                        discarded_reasons['excessive_jump'] += 1
                        continue

            # Update last valid weight and time
            last_valid_weight = weight_value
            last_valid_time = result['time_seconds']
            validated_results.append(result)

        except (ValueError, TypeError):
            # Keep frames with unclear/error readings
            validated_results.append(result)

    # Aggregate readings to 1-second intervals
    second_groups = {}
    for result in validated_results:
        whole_second = int(result['time_seconds'])  # 0.0->0, 0.5->0, 1.0->1, 1.5->1
        if whole_second not in second_groups:
            second_groups[whole_second] = []
        second_groups[whole_second].append(result)

    aggregated_results = []
    for second in sorted(second_groups.keys()):
        readings = second_groups[second]
        valid_weights = []

        for reading in readings:
            try:
                weight = float(reading['weight'])
                valid_weights.append(weight)
            except (ValueError, TypeError):
                # Skip 'unclear' or 'error' readings
                continue

        if not valid_weights:
            # No valid readings for this second, use first reading as-is for tracking
            aggregated_results.append({
                'frame': readings[0]['frame'],
                'time_seconds': second,
                'filename': readings[0]['filename'],
                'weight': readings[0]['weight']
            })
        elif len(valid_weights) == 1:
            # Only one valid reading, use it
            aggregated_results.append({
                'frame': readings[0]['frame'],
                'time_seconds': second,
                'filename': readings[0]['filename'],
                'weight': str(valid_weights[0])
            })
        else:
            # Multiple valid readings - check consistency and average
            weight_diff = max(valid_weights) - min(valid_weights)
            if weight_diff > MAX_INTRA_SECOND_DIFF:
                click.echo(f"\n  ⚠️  Second {second}: Large weight variance ({weight_diff:.1f}g) within same second")
                discarded_count += 1
                discarded_reasons['intra_second_diff'] += 1
                # Use the higher weight (more likely to be correct for accumulation)
                avg_weight = max(valid_weights)
            else:
                # Readings are consistent, use average
                avg_weight = sum(valid_weights) / len(valid_weights)

            aggregated_results.append({
                'frame': readings[0]['frame'],  # Use first frame of the second
                'time_seconds': second,
                'filename': f"averaged_second_{second}",
                'weight': str(avg_weight)
            })

    return aggregated_results, discarded_count

def process_all_frames(session_path=None, output_csv='weight_data.csv'):
    """Process all frame images and extract weights using async OCR"""
    
    # Check if API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        click.echo(click.style("Error: OPENAI_API_KEY environment variable not set!", fg='red'))
        click.echo("Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return None

    # Get all frame images from session directory
    if session_path:
        frames_dir = Path(session_path) / 'frames'
        frame_files = sorted(frames_dir.glob('frame_*.jpg'),
                            key=lambda x: int(x.stem.split('_')[1]))
        frame_files = [str(f) for f in frame_files]
        output_csv = str(Path(session_path) / output_csv)
    else:
        # Legacy: look in current directory
        frame_files = sorted(glob.glob('frame_*.jpg'),
                            key=lambda x: int(x.split('_')[1].split('.')[0]))

    if not frame_files:
        click.echo(click.style("No frame files found.", fg='red'))
        return None

    click.echo(f"Found {len(frame_files)} frames to process...")

    # Process frames asynchronously
    async def run_async_processing():
        processor = AsyncOCRProcessor(
            api_key=api_key,
            max_concurrent=get_max_concurrent()
        )

        # Process with progress bar
        results = []
        with click.progressbar(length=len(frame_files), label='Processing frames (async)') as bar:
            # Define progress callback
            def update_progress():
                bar.update(1)

            # Process all frames with progress updates
            results = await processor.process_all_frames_with_progress(
                frame_files,
                progress_callback=update_progress,
                max_per_second=get_max_per_second()
            )

        return results

    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        raw_results = loop.run_until_complete(run_async_processing())
    finally:
        loop.close()

    # Initialize tracking for validation
    discarded_reasons = {'decreasing': 0, 'excessive_jump': 0, 'intra_second_diff': 0}

    # Validate and aggregate results
    results, discarded_count = validate_and_aggregate_results(raw_results, discarded_reasons)
    
    # Save results to CSV
    with open(output_csv, 'w') as f:
        f.write('time_seconds,frame,filename,weight\n')
        for result in results:
            f.write(f"{result['time_seconds']},{result['frame']},{result['filename']},{result['weight']}\n")

    click.echo(click.style("\n✓ Processing complete!", fg='green'))
    click.echo(f"Results saved to {output_csv}")
    click.echo(f"Processed {len(raw_results)} frames at 2 fps → {len(results)} aggregated seconds")
    click.echo(f"Successfully recorded {len([r for r in results if r['weight'] not in ['error', 'unclear']])} valid seconds")
    if discarded_count > 0:
        click.echo(click.style(f"⚠️  Found {discarded_count} data quality issues:", fg='yellow'))
        if discarded_reasons['decreasing'] > 0:
            click.echo(f"    - {discarded_reasons['decreasing']} frames: weight decreased (transition error)")
        if discarded_reasons['excessive_jump'] > 0:
            click.echo(f"    - {discarded_reasons['excessive_jump']} frames: excessive jump >20ml/s (misread)")
        if discarded_reasons['intra_second_diff'] > 0:
            click.echo(f"    - {discarded_reasons['intra_second_diff']} seconds: high variance within same second")
    
    return output_csv

def create_uroflow_plot(csv_file='weight_data.csv', output_file='uroflow_chart.png', show_plot=False, metrics=None):
    """Create a comprehensive uroflow visualization chart with both raw and smoothed curves"""

    # If metrics provided, use that data (includes both raw and smoothed)
    if metrics and 'flow_rates' in metrics and 'smoothed_flow_rates' in metrics:
        times = metrics['times']
        weights = metrics['weights']
        flow_times = metrics['flow_times']
        flow_rates = metrics['flow_rates']
        smoothed_flow_rates = metrics['smoothed_flow_rates']
        peak_flow = metrics['peak_flow_rate']  # This is the smoothed Qmax
        peak_flow_index = metrics.get('peak_flow_index', 0)  # Get index from metrics
        avg_flow = metrics['average_flow_rate']  # This is also from smoothed data
    else:
        # Read the CSV data
        times = []
        weights = []

        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        weight = float(row['weight'])
                        time = float(row['time_seconds'])
                        times.append(time)
                        weights.append(weight)
                    except (ValueError, TypeError):
                        continue
        except FileNotFoundError:
            click.echo(click.style(f"Error: {csv_file} not found. Run 'uroflow read' first to process frames.", fg='red'))
            return None

        if len(weights) < 2:
            click.echo(click.style("Insufficient valid data points for plotting", fg='red'))
            return None

        # Calculate flow rates (ml/s)
        flow_rates = []
        flow_times = []

        for i in range(1, len(weights)):
            dt = times[i] - times[i-1]
            if dt > 0:
                dw = weights[i] - weights[i-1]
                flow_rate = max(0, dw / dt)  # Only positive flow rates
                flow_rates.append(flow_rate)
                flow_times.append(times[i])

        # Apply smoothing
        smoothed_flow_rates = smooth_flow_rates(flow_rates)

        # Find peak flow using 2-second rule (returns both value and index)
        # Pass already_smoothed=True since we're providing smoothed data
        peak_flow, peak_flow_index = calculate_qmax_2sec(flow_times, smoothed_flow_rates, already_smoothed=True)

        # Calculate average flow (excluding zero flows)
        non_zero_flows = [f for f in smoothed_flow_rates if f > MIN_FLOW_THRESHOLD]
        avg_flow = sum(non_zero_flows) / len(non_zero_flows) if non_zero_flows else 0

    # Calculate cumulative volume (ml)
    initial_weight = weights[0]
    volumes = [w - initial_weight for w in weights]

    # Calculate peak flow time using the index from calculate_qmax_2sec or metrics
    # peak_flow_index is already set either from metrics or from calculate_qmax_2sec
    peak_flow_time = flow_times[peak_flow_index] if flow_times and peak_flow_index < len(flow_times) else 0

    # Create figure with landscape A4 dimensions (11.7 x 8.3 inches)
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle('Uroflowmetry Analysis Report', fontsize=16, fontweight='bold')

    # Create gridspec for better layout control
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[3, 1])

    # Main plot - dual axis
    ax1 = fig.add_subplot(gs[0, :])

    # Plot cumulative volume on primary y-axis
    color1 = '#2E86AB'  # Nice blue
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Cumulative Volume (ml)', color=color1, fontsize=12)
    line1 = ax1.plot(times, volumes, color=color1, linewidth=2.5, label='Volume', marker='o', markersize=3)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Create second y-axis for flow rate
    ax2 = ax1.twinx()
    color2 = '#A23B72'  # Nice purple
    color2_raw = '#B87A7A'  # Darker purple for raw data (30% darker than #D4A5A5)
    ax2.set_ylabel('Flow Rate (ml/s)', color=color2, fontsize=12)

    # Plot both raw and smoothed flow rates
    line2_raw = ax2.plot(flow_times, flow_rates, color=color2_raw, linewidth=1, label='Raw Flow Rate', alpha=0.6, linestyle=':')
    smoothing_label = f'Smoothed Flow Rate ({SMOOTHING_SECONDS}s window, 2-sec rule)'
    line2_smoothed = ax2.plot(flow_times, smoothed_flow_rates, color=color2, linewidth=2.5, label=smoothing_label, alpha=0.9)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Fill area under smoothed flow rate curve
    ax2.fill_between(flow_times, smoothed_flow_rates, alpha=0.2, color=color2)

    # Mark Qmax with vertical line and annotation
    if peak_flow > 0:
        ax2.axvline(x=peak_flow_time, color='#F18F01', linestyle='--', linewidth=2, alpha=0.8)
        ax2.annotate(f'Qmax = {peak_flow:.1f} ml/s\n@ {peak_flow_time:.1f}s',
                    xy=(peak_flow_time, peak_flow),
                    xytext=(peak_flow_time + 2, peak_flow * 0.9),
                    fontsize=11,
                    fontweight='bold',
                    color='#F18F01',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#F18F01', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='#F18F01', lw=1.5))

    # Add average flow line
    if avg_flow > 0:
        ax2.axhline(y=avg_flow, color='#C73E1D', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Avg Flow = {avg_flow:.1f} ml/s')

    # Set axis limits
    ax1.set_xlim(0, max(times) * 1.05)
    ax1.set_ylim(0, max(volumes) * 1.1)
    # Use max of both raw and smoothed for y-axis limit
    max_flow = max(max(flow_rates) if flow_rates else 0, max(smoothed_flow_rates) if smoothed_flow_rates else 0)
    ax2.set_ylim(0, max_flow * 1.2)

    # Add legend with all lines
    lines = line1 + line2_raw + line2_smoothed
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)

    # Add title to main plot
    ax1.set_title('Flow Rate and Volume Over Time', fontsize=14, pad=20)

    # Statistics box
    stats_ax = fig.add_subplot(gs[1, 0])
    stats_ax.axis('off')

    # Calculate metrics
    total_volume = volumes[-1] if volumes else 0
    total_time = times[-1] if times else 0

    # Create stats text
    stats_text = (
        f"Key Metrics (Smoothed):\n\n"
        f"Total Volume: {total_volume:.1f} ml\n"
        f"Peak Flow Rate (Qmax)*: {peak_flow:.1f} ml/s\n"
        f"Average Flow Rate: {avg_flow:.1f} ml/s\n"
        f"Time to Peak: {peak_flow_time:.1f} s\n"
        f"Total Time: {total_time:.1f} s\n\n"
        f"Reference Values (Adult Male):\n"
        f"Normal Qmax: >15 ml/s\n"
        f"Normal Volume: 150-500 ml\n\n"
        f"*2-sec sustained rule applied"
    )

    stats_ax.text(0.05, 0.95, stats_text, transform=stats_ax.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='gray', alpha=0.8))

    # Clinical interpretation box
    interp_ax = fig.add_subplot(gs[1, 1])
    interp_ax.axis('off')

    # Determine interpretation
    interpretations = []
    colors = []

    if peak_flow < BORDERLINE_QMAX:
        interpretations.append("⚠ Low peak flow")
        colors.append('red')
    elif peak_flow < NORMAL_QMAX_MIN:
        interpretations.append("⚠ Borderline peak flow")
        colors.append('orange')
    else:
        interpretations.append("✓ Normal peak flow")
        colors.append('green')

    if total_volume < NORMAL_VOLUME_MIN:
        interpretations.append("⚠ Low volume")
        colors.append('orange')
    elif total_volume > NORMAL_VOLUME_MAX:
        interpretations.append("⚠ High volume")
        colors.append('orange')
    else:
        interpretations.append("✓ Normal volume")
        colors.append('green')

    if avg_flow < NORMAL_QAVE_MIN:
        interpretations.append("⚠ Low average flow")
        colors.append('red')
    else:
        interpretations.append("✓ Normal average flow")
        colors.append('green')

    # Create interpretation text with colors
    interp_text = "Clinical Notes:\n\n"
    for i, (text, color) in enumerate(zip(interpretations, colors)):
        interp_ax.text(0.05, 0.75 - i*0.15, text, transform=interp_ax.transAxes,
                      fontsize=10, color=color, fontweight='bold')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')

    if show_plot:
        plt.show()
    else:
        plt.close()

    click.echo(click.style(f"\n✓ Chart saved to {output_file}", fg='green'))
    return output_file

def analyze_uroflow_data(csv_file='weight_data.csv', create_plot=False, generate_report=False, session_path=None):
    """Analyze the weight data and calculate uroflowmetry metrics"""
    
    # Read the CSV data
    times = []
    weights = []
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    weight = float(row['weight'])
                    time = float(row['time_seconds'])
                    times.append(time)
                    weights.append(weight)
                except (ValueError, TypeError):
                    # Skip 'unclear' or 'error' readings
                    continue
    except FileNotFoundError:
        click.echo(click.style(f"Error: {csv_file} not found. Run 'uroflow read' first to process frames.", fg='red'))
        return None
    
    if len(weights) < 2:
        click.echo(click.style("Insufficient valid data points for analysis", fg='red'))
        return None
    
    # Calculate flow rates (ml/s) - derivative of weight over time
    flow_rates = []
    flow_times = []
    
    for i in range(1, len(weights)):
        dt = times[i] - times[i-1]
        if dt > 0:  # Avoid division by zero
            dw = weights[i] - weights[i-1]
            flow_rate = dw / dt  # ml/s (since 1g ≈ 1ml for urine)
            if flow_rate >= 0:  # Only include positive flow rates
                flow_rates.append(flow_rate)
                flow_times.append(times[i])
    
    if not flow_rates:
        click.echo(click.style("No valid flow rate data calculated", fg='red'))
        return None
    
    # Find start of urination (first significant weight increase)
    start_index = 0
    for i in range(1, len(weights)):
        if weights[i] - weights[0] > MIN_WEIGHT_CHANGE_THRESHOLD:
            start_index = i - 1
            break
    
    # Calculate metrics
    initial_weight = weights[0]
    final_weight = weights[-1]
    voided_volume = final_weight - initial_weight
    
    # Time metrics
    time_to_start = times[start_index] if start_index > 0 else 0
    
    # Flow metrics
    # Use smoothed flow rates for analysis
    smoothed_flow_rates = smooth_flow_rates(flow_rates)

    # Calculate Qmax using 2-second rule (clinically accurate)
    # Pass already_smoothed=True since we're providing smoothed data
    peak_flow_rate, peak_flow_index = calculate_qmax_2sec(flow_times, smoothed_flow_rates, already_smoothed=True)

    # Calculate time to peak
    time_to_peak = flow_times[peak_flow_index] - time_to_start if peak_flow_index < len(flow_times) else 0
    
    # Calculate average flow rate (excluding zero flow periods, using smoothed data)
    non_zero_flows = [f for f in smoothed_flow_rates if f > MIN_FLOW_THRESHOLD]
    average_flow_rate = sum(non_zero_flows) / len(non_zero_flows) if non_zero_flows else 0
    
    # Flow time (from start to when flow essentially stops)
    flow_end_index = len(weights) - 1
    for i in range(len(weights) - 1, start_index, -1):
        if weights[i] - weights[i-1] > MIN_FLOW_THRESHOLD:
            flow_end_index = i
            break
    
    flow_time = times[flow_end_index] - times[start_index]
    
    # Display results
    click.echo("\n" + "="*60)
    click.echo(click.style("UROFLOWMETRY ANALYSIS", fg='cyan', bold=True))
    click.echo("="*60)
    
    click.echo("\n📊 KEY METRICS:")
    click.echo("-" * 40)
    click.echo(f"Voided Volume:         {voided_volume:.1f} ml")
    click.echo(f"Peak Flow Rate (Qmax): {peak_flow_rate:.1f} ml/s")
    click.echo(f"Average Flow Rate (Qave): {average_flow_rate:.1f} ml/s")
    click.echo(f"Time to Start:         {time_to_start:.1f} seconds")
    click.echo(f"Time to Peak:          {time_to_peak:.1f} seconds")
    click.echo(f"Flow Time:             {flow_time:.1f} seconds")
    click.echo(f"Voiding Time:          {times[-1]:.1f} seconds (total)")
    
    # Clinical interpretation hints
    click.echo("\n📋 REFERENCE VALUES (adult male):")
    click.echo("-" * 40)
    click.echo(f"Normal Qmax:           > {NORMAL_QMAX_MIN:.0f} ml/s")
    click.echo(f"Normal Qave:           > {NORMAL_QAVE_MIN:.0f} ml/s")
    click.echo(f"Normal volume:         {NORMAL_VOLUME_MIN:.0f}-{NORMAL_VOLUME_MAX:.0f} ml")
    
    # Simple interpretation
    click.echo("\n💡 OBSERVATIONS:")
    click.echo("-" * 40)
    
    if peak_flow_rate < BORDERLINE_QMAX:
        click.echo(click.style("⚠️  Peak flow rate is below normal range", fg='red'))
    elif peak_flow_rate < NORMAL_QMAX_MIN:
        click.echo(click.style("⚠️  Peak flow rate is borderline", fg='yellow'))
    else:
        click.echo(click.style("✓ Peak flow rate is within normal range", fg='green'))
    
    if voided_volume < NORMAL_VOLUME_MIN:
        click.echo(click.style("⚠️  Low voided volume - may affect accuracy", fg='yellow'))
    elif voided_volume > NORMAL_VOLUME_MAX:
        click.echo(click.style("⚠️  High voided volume", fg='yellow'))
    else:
        click.echo(click.style("✓ Voided volume is within normal range", fg='green'))
    
    if average_flow_rate < NORMAL_QAVE_MIN:
        click.echo(click.style("⚠️  Average flow rate is below normal", fg='red'))
    else:
        click.echo(click.style("✓ Average flow rate is within normal range", fg='green'))
    
    click.echo("\n" + "="*60)

    # Prepare metrics dictionary
    metrics = {
        'voided_volume': voided_volume,
        'peak_flow_rate': peak_flow_rate,
        'peak_flow_index': peak_flow_index,  # Include index for chart alignment
        'average_flow_rate': average_flow_rate,
        'time_to_start': time_to_start,
        'time_to_peak': time_to_peak,
        'flow_time': flow_time,
        'total_time': times[-1],
        # Include flow data for charting
        'times': times,
        'weights': weights,
        'flow_times': flow_times,
        'flow_rates': flow_rates,
        'smoothed_flow_rates': smoothed_flow_rates
    }

    # Create plot if requested
    if create_plot:
        # Determine output path based on CSV location
        csv_path = Path(csv_file)
        if csv_path.parent.name == 'sessions' or 'uroflow/sessions' in str(csv_path):
            # We're in a session directory
            output_file = csv_path.parent / 'uroflow_chart.png'
            if not session_path:
                session_path = csv_path.parent
        else:
            output_file = 'uroflow_chart.png'
        create_uroflow_plot(csv_file, str(output_file), metrics=metrics)

    # Generate PDF report if requested
    if generate_report and session_path:
        report_generator = UroflowReportGenerator(Path(session_path))
        report_path = report_generator.generate_report(metrics)
        click.echo(click.style(f"\n📄 PDF report generated: {report_path}", fg='green'))

    return metrics

@click.group()
def cli():
    """DIY Uroflowmetry Analysis Tool - Extract flow metrics from scale video frames"""
    pass

@cli.command()
@click.option('--output-csv', default='weight_data.csv', help='Output CSV filename')
@click.option('--session', help='Session ID or path to use')
@click.option('--patient-name', help='Patient name for new session')
@click.option('--force', is_flag=True, help='Force re-processing, ignore cached OCR data')
def read(output_csv, session, patient_name, force):
    """Process frame images and extract weight readings using OpenAI Vision API"""
    session_mgr = SessionManager()

    # Get or create session
    if session:
        session_path = session_mgr.get_session(session)
        if not session_path:
            click.echo(click.style(f"Session '{session}' not found", fg='red'))
            return
    else:
        # Create new session or use latest
        session_path = session_mgr.get_session('latest')
        if not session_path:
            session_path = session_mgr.create_session(patient_name)
            click.echo(f"Created new session: {session_path.name}")

    # Force re-processing if requested
    if force:
        click.echo("Force mode: Removing existing OCR data...")
        csv_path = session_path / output_csv
        if csv_path.exists():
            csv_path.unlink()
            click.echo(f"  Removed {csv_path.name}")

    # Check if OCR is needed
    if not force and not session_mgr.should_run_ocr(session_path):
        click.echo(click.style("✓ OCR already completed for this session, skipping... (use --force to re-process)", fg='green'))
        csv_file = session_path / output_csv
    else:
        csv_file = process_all_frames(session_path, output_csv)

    if csv_file:
        click.echo("\nRunning analysis on the extracted data...")
        analyze_uroflow_data(str(csv_file), create_plot=True, generate_report=False, session_path=session_path)

@cli.command()
@click.option('--csv-file', help='CSV file to analyze (default: latest session)')
@click.option('--plot/--no-plot', default=True, help='Generate visualization chart')
@click.option('--session', help='Session ID to analyze')
@click.option('--smoothing', type=float, help='Smoothing window in seconds (default: 6.0)')
def analyze(csv_file, plot, session, smoothing):
    """Analyze existing CSV data and display uroflowmetry metrics"""
    # Override smoothing if specified
    if smoothing:
        global SMOOTHING_WINDOW_SIZE, SMOOTHING_SECONDS
        SMOOTHING_SECONDS = smoothing
        SMOOTHING_WINDOW_SIZE = int(smoothing * 2)  # Convert to points at 2 fps
        click.echo(f"Using {smoothing}-second smoothing window ({SMOOTHING_WINDOW_SIZE} points)")

    if csv_file:
        # Direct file path provided
        analyze_uroflow_data(csv_file, create_plot=plot)
    else:
        # Use session
        session_mgr = SessionManager()
        session_path = session_mgr.get_session(session or 'latest')

        if not session_path:
            click.echo(click.style("No sessions found. Run 'uroflow read' first.", fg='red'))
            return

        csv_path = session_path / 'weight_data.csv'
        if not csv_path.exists():
            click.echo(click.style(f"No data found in session. Run 'uroflow read' first.", fg='red'))
            return

        analyze_uroflow_data(str(csv_path), create_plot=plot, generate_report=False, session_path=session_path)

@cli.command()
@click.option('--csv-file', help='CSV file to plot (default: latest session)')
@click.option('--output', help='Output PNG filename')
@click.option('--show/--no-show', default=False, help='Display plot interactively')
@click.option('--session', help='Session ID to plot')
@click.option('--smoothing', type=float, help='Smoothing window in seconds (default: 6.0)')
def plot(csv_file, output, show, session, smoothing):
    """Create a visualization chart from uroflow data"""
    # Override smoothing if specified
    if smoothing:
        global SMOOTHING_WINDOW_SIZE, SMOOTHING_SECONDS
        SMOOTHING_SECONDS = smoothing
        SMOOTHING_WINDOW_SIZE = int(smoothing * 2)  # Convert to points at 2 fps
        click.echo(f"Using {smoothing}-second smoothing window ({SMOOTHING_WINDOW_SIZE} points)")

    if csv_file:
        # Direct file path provided
        output = output or 'uroflow_chart.png'
        create_uroflow_plot(csv_file, output, show_plot=show)
    else:
        # Use session
        session_mgr = SessionManager()
        session_path = session_mgr.get_session(session or 'latest')

        if not session_path:
            click.echo(click.style("No sessions found. Run 'uroflow read' first.", fg='red'))
            return

        csv_path = session_path / 'weight_data.csv'
        if not csv_path.exists():
            click.echo(click.style(f"No data found in session. Run 'uroflow read' first.", fg='red'))
            return

        output_path = session_path / (output or 'uroflow_chart.png')
        create_uroflow_plot(str(csv_path), str(output_path), show_plot=show)

@cli.command()
def sessions():
    """List all analysis sessions"""
    session_mgr = SessionManager()
    sessions = session_mgr.list_sessions()

    if not sessions:
        click.echo("No sessions found.")
        return

    click.echo("\nAvailable sessions:")
    click.echo("="*80)

    for sess in sessions:
        patient_info = f" - {sess['patient_name']}" if sess['patient_name'] else ""
        steps = sess['steps']

        # Create detailed status information
        status_parts = []
        if steps.get('frames_extracted'):
            status_parts.append('✓ Frames')
        else:
            status_parts.append('○ Frames')

        if steps.get('ocr_completed'):
            status_parts.append('✓ OCR')
        else:
            status_parts.append('○ OCR')

        if steps.get('analysis_completed'):
            status_parts.append('✓ Analysis')
        else:
            status_parts.append('○ Analysis')

        if steps.get('report_generated'):
            status_parts.append('✓ Report')
        else:
            status_parts.append('○ Report')

        status_str = ' | '.join(status_parts)

        click.echo(f"{sess['id']}{patient_info}")
        click.echo(f"  Status: {status_str}")
        click.echo(f"  Path: {sess['path']}")
        click.echo("-"*80)

@cli.command()
@click.option('--session', help='Session ID to generate report for (default: latest)')
def report(session):
    """Generate or regenerate PDF report for a session"""
    session_mgr = SessionManager()
    session_path = session_mgr.get_session(session or 'latest')

    if not session_path:
        click.echo(click.style("No sessions found.", fg='red'))
        return

    # Check if we have the necessary files
    csv_path = session_path / 'weight_data.csv'
    if not csv_path.exists():
        click.echo(click.style(f"No analysis data found in session. Run 'uroflow analyze' first.", fg='red'))
        return

    chart_path = session_path / 'uroflow_chart.png'
    if not chart_path.exists():
        click.echo("Chart not found, generating...")
        create_uroflow_plot(str(csv_path), str(chart_path))

    # Analyze data to get metrics
    click.echo(f"Generating report for session: {session_path.name}")
    metrics = analyze_uroflow_data(str(csv_path), create_plot=False, generate_report=False)

    # Generate PDF report
    report_generator = UroflowReportGenerator(session_path)
    report_path = report_generator.generate_report(metrics)

    click.echo(click.style(f"✅ Report generated: {report_path}", fg='green', bold=True))

@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--patient-name', prompt='Patient name (optional)', default='', help='Patient name for the session')
@click.option('--fps', default=2, help='Frames per second to extract')
@click.option('--force', is_flag=True, help='Force re-processing, ignore all cached data')
def process(video_path, patient_name, fps, force):
    """Process a video file from start to finish (frames -> OCR -> analysis -> report)"""
    session_mgr = SessionManager()
    video_path = Path(video_path).resolve()

    # Clean patient name
    patient_name = patient_name.strip() if patient_name else None

    # Get or create session for this video
    session_path = session_mgr.get_or_create_session_from_video(video_path, patient_name)
    click.echo(f"\nUsing session: {session_path.name}")

    # Force re-processing if requested
    if force:
        click.echo("Force mode: Cleaning existing data...")
        # Remove frames
        frames_dir = session_path / 'frames'
        if frames_dir.exists():
            for frame in frames_dir.glob('frame_*.jpg'):
                frame.unlink()
            click.echo(f"  Removed frame images")
        # Remove OCR data
        csv_path = session_path / 'weight_data.csv'
        if csv_path.exists():
            csv_path.unlink()
            click.echo(f"  Removed weight_data.csv")
        # Remove chart
        chart_path = session_path / 'uroflow_chart.png'
        if chart_path.exists():
            chart_path.unlink()
            click.echo(f"  Removed chart")

    # Step 1: Extract frames if needed
    if force or session_mgr.should_extract_frames(session_path, video_path):
        click.echo("\n📷 Extracting frames from video...")
        frames_dir = session_path / 'frames'

        # Check if ffmpeg is available
        if not shutil.which('ffmpeg'):
            click.echo(click.style("Error: ffmpeg not found. Please install ffmpeg.", fg='red'))
            click.echo("On macOS: brew install ffmpeg")
            return

        # Run ffmpeg to extract frames
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', f'fps={fps}',
            '-q:v', '2',
            str(frames_dir / 'frame_%04d.jpg')
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                click.echo(click.style(f"Error extracting frames: {result.stderr}", fg='red'))
                return

            # Count extracted frames
            frame_count = len(list(frames_dir.glob('frame_*.jpg')))
            click.echo(click.style(f"✓ Extracted {frame_count} frames", fg='green'))
        except Exception as e:
            click.echo(click.style(f"Error running ffmpeg: {e}", fg='red'))
            return
    else:
        click.echo(click.style("✓ Frames already extracted, using cached frames", fg='green'))

    # Step 2: Run OCR if needed
    if force or session_mgr.should_run_ocr(session_path):
        click.echo("\n🔍 Processing frames with OCR...")
        csv_file = process_all_frames(session_path)
        if not csv_file:
            click.echo(click.style("Error during OCR processing", fg='red'))
            return
    else:
        click.echo(click.style("✓ OCR already completed, using cached data", fg='green'))

    # Step 3: Analyze and create chart
    click.echo("\n📊 Analyzing uroflow data...")
    csv_path = session_path / 'weight_data.csv'
    analyze_uroflow_data(str(csv_path), create_plot=True, generate_report=True, session_path=session_path)

    click.echo(click.style(f"\n✅ Processing complete! Session: {session_path.name}", fg='green', bold=True))
    click.echo(f"Results saved in: {session_path}")

if __name__ == '__main__':
    cli()