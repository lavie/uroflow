#!/usr/bin/env python3
import os
import base64
import json
import glob
from openai import OpenAI
import time
import csv

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
            model="gpt-4o",
            max_tokens=100,
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

def process_all_frames():
    """Process all frame images and extract weights"""
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set!")
        print("Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Get all frame images, sorted by frame number
    frame_files = sorted(glob.glob('frame_*.jpg'), 
                        key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if not frame_files:
        print("No frame files found. Make sure you have frame_*.jpg files in the current directory.")
        return
    
    print(f"Found {len(frame_files)} frames to process...")
    
    results = []
    last_valid_weight = 0.0
    discarded_count = 0
    
    for i, frame_file in enumerate(frame_files):
        print(f"Processing {frame_file} ({i+1}/{len(frame_files)})...")
        
        weight = extract_weight_from_image(frame_file)
        frame_number = int(frame_file.split('_')[1].split('.')[0])
        
        # Try to convert weight to float for validation
        try:
            weight_value = float(weight)
            
            # Check for non-decreasing constraint
            if weight_value < last_valid_weight:
                print(f"  ⚠️  WARNING: Weight decreased from {last_valid_weight} to {weight_value}")
                print(f"     Likely a display transition error - discarding this frame")
                discarded_count += 1
                # Skip adding this result
                time.sleep(0.5)
                continue
            
            # Update last valid weight
            last_valid_weight = weight_value
            
        except (ValueError, TypeError):
            # Weight is 'unclear' or 'error' - keep it in results for tracking
            pass
        
        result = {
            'frame': frame_number,
            'time_seconds': frame_number - 1,  # Frame 1 = 0 seconds, Frame 2 = 1 second, etc.
            'filename': frame_file,
            'weight': weight
        }
        
        results.append(result)
        print(f"  -> Weight: {weight}")
        
        # Be nice to the API - small delay between requests
        time.sleep(0.5)
    
    # Save results to CSV
    with open('weight_data.csv', 'w') as f:
        f.write('time_seconds,frame,filename,weight\n')
        for result in results:
            f.write(f"{result['time_seconds']},{result['frame']},{result['filename']},{result['weight']}\n")
    
    # Also save as JSON for more detailed analysis
    with open('weight_data.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Results saved to weight_data.csv and weight_data.json")
    print(f"Successfully processed {len([r for r in results if r['weight'] not in ['error', 'unclear']])} frames")
    if discarded_count > 0:
        print(f"⚠️  Discarded {discarded_count} frames due to decreasing weight (display transition errors)")
    
    return 'weight_data.csv'

def analyze_uroflow_data(csv_file='weight_data.csv'):
    """Analyze the weight data and calculate uroflowmetry metrics"""
    
    print("\n" + "="*60)
    print("UROFLOWMETRY ANALYSIS")
    print("="*60)
    
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
        print(f"Error: {csv_file} not found. Run the frame processing first.")
        return
    
    if len(weights) < 2:
        print("Insufficient valid data points for analysis")
        return
    
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
        print("No valid flow rate data calculated")
        return
    
    # Find start of urination (first significant weight increase)
    start_index = 0
    for i in range(1, len(weights)):
        if weights[i] - weights[0] > 0.5:  # 0.5g threshold for start
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
    non_zero_flows = [f for f in flow_rates if f > 0.1]
    average_flow_rate = sum(non_zero_flows) / len(non_zero_flows) if non_zero_flows else 0
    
    # Flow time (from start to when flow essentially stops)
    flow_end_index = len(weights) - 1
    for i in range(len(weights) - 1, start_index, -1):
        if weights[i] - weights[i-1] > 0.1:  # Last significant flow
            flow_end_index = i
            break
    
    flow_time = times[flow_end_index] - times[start_index]
    
    # Display results
    print("\n📊 KEY METRICS:")
    print("-" * 40)
    print(f"Voided Volume:        {voided_volume:.1f} ml")
    print(f"Peak Flow Rate (Qmax): {peak_flow_rate:.1f} ml/s")
    print(f"Average Flow Rate (Qave): {average_flow_rate:.1f} ml/s")
    print(f"Time to Start:        {time_to_start:.1f} seconds")
    print(f"Time to Peak:         {time_to_peak:.1f} seconds")
    print(f"Flow Time:            {flow_time:.1f} seconds")
    print(f"Voiding Time:         {times[-1]:.1f} seconds (total)")
    
    # Clinical interpretation hints
    print("\n📋 REFERENCE VALUES (adult male):")
    print("-" * 40)
    print("Normal Qmax:          > 15 ml/s")
    print("Normal Qave:          > 10 ml/s")
    print("Normal volume:        150-500 ml")
    
    # Simple interpretation
    print("\n💡 OBSERVATIONS:")
    print("-" * 40)
    
    if peak_flow_rate < 10:
        print("⚠️  Peak flow rate is below normal range")
    elif peak_flow_rate < 15:
        print("⚠️  Peak flow rate is borderline")
    else:
        print("✓ Peak flow rate is within normal range")
    
    if voided_volume < 150:
        print("⚠️  Low voided volume - may affect accuracy")
    elif voided_volume > 500:
        print("⚠️  High voided volume")
    else:
        print("✓ Voided volume is within normal range")
    
    if average_flow_rate < 10:
        print("⚠️  Average flow rate is below normal")
    else:
        print("✓ Average flow rate is within normal range")
    
    print("\n" + "="*60)
    
    return {
        'voided_volume': voided_volume,
        'peak_flow_rate': peak_flow_rate,
        'average_flow_rate': average_flow_rate,
        'time_to_start': time_to_start,
        'time_to_peak': time_to_peak,
        'flow_time': flow_time,
        'total_time': times[-1]
    }

if __name__ == "__main__":
    csv_file = process_all_frames()
    if csv_file:
        analyze_uroflow_data(csv_file)
