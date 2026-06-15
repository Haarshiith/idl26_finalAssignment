import json
import subprocess
import time
import sys

# Optimal architecture pairings based on dataset characteristics and preliminary experiments
pairings = [
    {"DATA": "cells", "MODEL": "AlexNet", "EPOCHS": 10, "LEARNING_RATE": 0.001},
    {"DATA": "chest", "MODEL": "ResNet18", "EPOCHS": 15, "LEARNING_RATE": 0.001},
    {"DATA": "lesions", "MODEL": "VGG16", "EPOCHS": 15, "LEARNING_RATE": 0.0005},
    {"DATA": "orgs", "MODEL": "ResNet18", "EPOCHS": 20, "LEARNING_RATE": 0.0002}
]

# Base configuration that applies to all benchmarks, with specific pairings overriding as needed
base_config = {
    "DATA_PATH": "data",
    "BATCH_SIZE": 16,
    "CHANNELS": 1,
    "NUM_CLASSES": 11,
    "ACTIVATION": "ReLU"
}

def main():
    print("="*60)
    print("INITIATING AUTOMATED CONSOLIDATED BENCHMARK SUITE")
    print("="*60 + "\n")

    start_time = time.time()

    for pair in pairings:
        print(f"\n>>> Preparing Environment for: {pair['DATA'].upper()} paired with {pair['MODEL']} <<<")
        
        # Merge the specific pairing with the base config
        current_config = {**base_config, **pair}
        
        # Write the temporary config to config.json
        with open("config.json", "w") as f:
            json.dump(current_config, f, indent=4)
            
        print(f"Config locked. Booting isolated training process...")

        script_to_run = "Code/transfer.py" if pair['DATA'] == "orgs" else "Code/train.py"
        
        # Execute as a separate process to ensure clean GPU VRAM between runs
        try:
            subprocess.run([sys.executable, script_to_run], check=True)
        except subprocess.CalledProcessError:
            print(f"!!! CRITICAL FAILURE during {pair['DATA']} execution. Halting suite. !!!")
            break
            
        print(f"\n[SUCCESS] {pair['DATA']} benchmark complete.")
        print("-" * 60)

    total_time = (time.time() - start_time) / 60
    print(f"\nALL BENCHMARKS COMPLETE. Total Suite Execution Time: {total_time:.2f} minutes.")

if __name__ == "__main__":
    main()