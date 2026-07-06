import json
import subprocess
import time
import sys
import re

# Optimal architecture pairings, now with dynamic Channels, Classes and Pretraining.
pairings = [
    # 1. Standard Baseline
    {"DATA": "cells", "MODEL": "AlexNet", "EPOCHS": 15, "LEARNING_RATE": 0.001, "CHANNELS": 3, "NUM_CLASSES": 8, "MODE": "train"},

    # 2. Standard Baseline + The Green Initiative (Head-to-Head Comparison on 'chest')
    {"DATA": "chest", "MODEL": "ResNet18", "EPOCHS": 20, "LEARNING_RATE": 0.0005, "CHANNELS": 1, "NUM_CLASSES": 2, "MODE": "train"},
    {"DATA": "chest", "MODEL": "SlimResNet", "EPOCHS": 20, "LEARNING_RATE": 0.0005, "CHANNELS": 1, "NUM_CLASSES": 2, "MODE": "train"},
    
    # 3. Standard Baselines
    {"DATA": "lesions", "MODEL": "VGG16", "EPOCHS": 20, "LEARNING_RATE": 0.00005, "CHANNELS": 3, "NUM_CLASSES": 7, "MODE": "train"},
    {"DATA": "orgs", "MODEL": "ResNet18", "EPOCHS": 20, "LEARNING_RATE": 0.0002, "CHANNELS": 1, "NUM_CLASSES": 11, "MODE": "train"},
    
    # 4. The Data-Scarcity Task (Controlled Experiment: Equal LR, Equal Epochs)
    # Arm A: True Scratch
    {"DATA": "organs", "MODEL": "ResNet18", "EPOCHS": 20, "LEARNING_RATE": 0.0001, "CHANNELS": 1, "NUM_CLASSES": 11, "MODE": "train", "PRETRAINED": False}, 
    # Arm B: ImageNet Init Only
    {"DATA": "organs", "MODEL": "ResNet18", "EPOCHS": 20, "LEARNING_RATE": 0.0001, "CHANNELS": 1, "NUM_CLASSES": 11, "MODE": "train", "PRETRAINED": True},
    # Arm C: Full Transfer
    {"DATA": "organs", "MODEL": "ResNet18", "EPOCHS": 20, "LEARNING_RATE": 0.0001, "CHANNELS": 1, "NUM_CLASSES": 11, "MODE": "transfer"} 
]

base_config = {
    "DATA_PATH": "data",
    "BATCH_SIZE": 16,
    "ACTIVATION": "ReLU"
}

def main():
    print("="*60)
    print("INITIATING AUTOMATED CONSOLIDATED BENCHMARK SUITE")
    print("="*60 + "\n")

    start_time = time.time()
    
    # Prepare the Green Initiative Efficiency Matrix
    efficiency_log = "## Efficiency Verification Matrix\n\n| Dataset | Model | Mode | Training Time (s) | Inference Latency (s/sample) | Peak GPU Memory (Training) (MB) |\n|---|---|---|---|---|---|\n"

    for pair in pairings:
        mode_label = "TRANSFER" if pair['MODE'] == "transfer" else ("SCRATCH" if not pair.get("PRETRAINED", True) else "PRETRAINED")
        print(f"\n>>> Preparing Environment for: {pair['DATA'].upper()} ({mode_label}) paired with {pair['MODEL']} <<<")
        
        current_config = {**base_config, **pair}
        
        with open("config.json", "w") as f:
            json.dump(current_config, f, indent=4)

        script_to_run = "Code/transfer.py" if pair['MODE'] == "transfer" else "Code/train.py"
        
        try:
            # Capture output to extract Green Initiative metrics
            result = subprocess.run([sys.executable, script_to_run], capture_output=True, text=True, check=True)
            print(result.stdout)
            
            # Extract metrics using regex from the fit.py print statements
            matches = re.findall(r"Total Training Time:\s*([\d.]+)", result.stdout)
            i_lat = re.search(r"Inference Latency:\s*([\d.]+)", result.stdout)
            p_mem = re.search(r"Peak GPU Memory:\s*([\d.]+)", result.stdout)
            
            val_time = matches[-1] if matches else "N/A"
            val_lat = i_lat.group(1) if i_lat else "N/A"
            val_mem = p_mem.group(1) if p_mem else "0.00"
            
            # Append to our matrix
            efficiency_log += f"| {pair['DATA']} | {pair['MODEL']} | {mode_label} | {val_time} | {val_lat} | {val_mem} |\n"

        except subprocess.CalledProcessError as e:
            print(f"!!! CRITICAL FAILURE during {pair['DATA']} execution !!!")
            print(e.stderr)
            break
            
        print(f"\n[SUCCESS] {pair['DATA']} ({mode_label}) benchmark complete.")
        print("-" * 60)

    # Save the Green Initiative matrix to disk
    with open("EFFICIENCY_MATRIX.md", "w") as f:
        f.write(efficiency_log)

    total_time = (time.time() - start_time) / 60
    print(f"\nALL BENCHMARKS COMPLETE. Total Suite Execution Time: {total_time:.2f} minutes.")
    print("Efficiency metrics saved to EFFICIENCY_MATRIX.md")

if __name__ == "__main__":
    main()