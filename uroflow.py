#!/usr/bin/env python3
import os
import base64
import json
import glob
from openai import OpenAI
import time
import csv
import click

# Constants for validation and analysis
MAX_FLOW_RATE = 20.0  # ml/s - maximum physiologically realistic flow rate
MIN_WEIGHT_CHANGE_THRESHOLD = 0.5  # grams - minimum change to detect start of urination
MIN_FLOW_THRESHOLD = 0.1  # ml/s - minimum flow to consider as active voiding
API_DELAY = 0.5  # seconds - delay between API calls to avoid rate limiting

# Clinical reference values (adult male)
NORMAL_QMAX_MIN = 15.0  # ml/s - minimum normal peak flow
NORMAL_QAVE_MIN = 10.0  # ml/s - minimum normal average flow
NORMAL_VOLUME_MIN = 150.0  # ml - minimum normal voided volume
NORMAL_VOLUME_MAX = 500.0  # ml - maximum normal voided volume
BORDERLINE_QMAX = 10.0  # ml/s - borderline peak flow threshold

# OpenAI API configuration
VISION_MODEL = "gpt-4o"
MAX_TOKENS = 100

# Set up the client using environment variable
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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

def process_all_frames(output_csv='weight_data.csv', output_json='weight_data.json'):
    """Process all frame images and extract weights"""
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        click.echo(click.style("Error: OPENAI_API_KEY environment variable not set!", fg='red'))
        click.echo("Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return None
    
    # Get all frame images, sorted by frame number
    frame_files = sorted(glob.glob('frame_*.jpg'), 
                        key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if not frame_files:
        click.echo(click.style("No frame files found. Make sure you have frame_*.jpg files in the current directory.", fg='red'))
        return None
    
    click.echo(f"Found {len(frame_files)} frames to process...")
    
    results = []
    last_valid_weight = 0.0
    last_valid_time = -1
    discarded_count = 0
    discarded_reasons = {'decreasing': 0, 'excessive_jump': 0}
    
    with click.progressbar(frame_files, label='Processing frames') as bar:
        for frame_file in bar:
            weight = extract_weight_from_image(frame_file)
            frame_number = int(frame_file.split('_')[1].split('.')[0])
            current_time = frame_number - 1  # Frame 1 = 0 seconds
            
            # Try to convert weight to float for validation
            try:
                weight_value = float(weight)
                
                # Check for non-decreasing constraint
                if weight_value < last_valid_weight:
                    click.echo(f"\n  ⚠️  Frame {frame_file}: Weight decreased from {last_valid_weight} to {weight_value} - discarding")
                    discarded_count += 1
                    discarded_reasons['decreasing'] += 1
                    time.sleep(API_DELAY)
                    continue
                
                # Check for unrealistic jumps (>20g/s flow rate)
                if last_valid_time >= 0:
                    time_diff = current_time - last_valid_time
                    if time_diff > 0:
                        weight_diff = weight_value - last_valid_weight
                        flow_rate = weight_diff / time_diff
                        if flow_rate > MAX_FLOW_RATE:
                            click.echo(f"\n  ⚠️  Frame {frame_file}: Unrealistic flow rate of {flow_rate:.1f} ml/s - discarding")
                            discarded_count += 1
                            discarded_reasons['excessive_jump'] += 1
                            time.sleep(API_DELAY)
                            continue
                
                # Update last valid weight and time
                last_valid_weight = weight_value
                last_valid_time = current_time
                
            except (ValueError, TypeError):
                # Weight is 'unclear' or 'error' - keep it in results for tracking
                pass
            
            result = {
                'frame': frame_number,
                'time_seconds': frame_number - 1,
                'filename': frame_file,
                'weight': weight
            }
            
            results.append(result)
            
            # Be nice to the API - small delay between requests
            time.sleep(API_DELAY)
    
    # Save results to CSV
    with open(output_csv, 'w') as f:
        f.write('time_seconds,frame,filename,weight\n')
        for result in results:
            f.write(f"{result['time_seconds']},{result['frame']},{result['filename']},{result['weight']}\n")
    
    # Also save as JSON for more detailed analysis
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    click.echo(click.style("\n✓ Processing complete!", fg='green'))
    click.echo(f"Results saved to {output_csv} and {output_json}")
    click.echo(f"Successfully processed {len([r for r in results if r['weight'] not in ['error', 'unclear']])} frames")
    if discarded_count > 0:
        click.echo(click.style(f"⚠️  Discarded {discarded_count} frames due to display reading errors:", fg='yellow'))
        if discarded_reasons['decreasing'] > 0:
            click.echo(f"    - {discarded_reasons['decreasing']} frames: weight decreased (transition error)")
        if discarded_reasons['excessive_jump'] > 0:
            click.echo(f"    - {discarded_reasons['excessive_jump']} frames: excessive jump >20ml/s (misread)")
    
    return output_csv

def analyze_uroflow_data(csv_file='weight_data.csv'):
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
    peak_flow_rate = max(flow_rates) if flow_rates else 0
    peak_flow_index = flow_rates.index(peak_flow_rate) if flow_rates else 0
    time_to_peak = flow_times[peak_flow_index] - time_to_start if flow_rates else 0
    
    # Calculate average flow rate (excluding zero flow periods)
    non_zero_flows = [f for f in flow_rates if f > MIN_FLOW_THRESHOLD]
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
    
    return {
        'voided_volume': voided_volume,
        'peak_flow_rate': peak_flow_rate,
        'average_flow_rate': average_flow_rate,
        'time_to_start': time_to_start,
        'time_to_peak': time_to_peak,
        'flow_time': flow_time,
        'total_time': times[-1]
    }

@click.group()
def cli():
    """DIY Uroflowmetry Analysis Tool - Extract flow metrics from scale video frames"""
    pass

@cli.command()
@click.option('--output-csv', default='weight_data.csv', help='Output CSV filename')
@click.option('--output-json', default='weight_data.json', help='Output JSON filename')
def read(output_csv, output_json):
    """Process frame images and extract weight readings using OpenAI Vision API"""
    csv_file = process_all_frames(output_csv, output_json)
    if csv_file:
        click.echo("\nRunning analysis on the extracted data...")
        analyze_uroflow_data(csv_file)

@cli.command()
@click.option('--csv-file', default='weight_data.csv', help='CSV file to analyze')
def analyze(csv_file):
    """Analyze existing CSV data and display uroflowmetry metrics"""
    analyze_uroflow_data(csv_file)

if __name__ == '__main__':
    cli()