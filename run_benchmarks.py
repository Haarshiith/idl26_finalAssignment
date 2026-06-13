import json
import subprocess
import time
import sys

# 1. Define your master theory of optimal pairings
pairings = [
    {"DATA": "organs", "MODEL": "ResNet18", "EPOCHS": 15, "LEARNING_RATE": 0.0002}
    # {"DATA": "chest", "MODEL": "ResNet18", "EPOCHS": 30, "LEARNING_RATE": 0.001},
    # {"DATA": "cells", "MODEL": "AlexNet", "EPOCHS": 15, "LEARNING_RATE": 0.001},
    # {"DATA": "lesions", "MODEL": "VGG16", "EPOCHS": 30, "LEARNING_RATE": 0.0005}
]

# 2. Define the static pipeline constants
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
        
        # Execute the training script as a separate process to ensure clean GPU VRAM
        try:
            subprocess.run([sys.executable, "Code/train.py"], check=True)
        except subprocess.CalledProcessError:
            print(f"!!! CRITICAL FAILURE during {pair['DATA']} execution. Halting suite. !!!")
            break
            
        print(f"\n[SUCCESS] {pair['DATA']} benchmark complete.")
        print("-" * 60)

    total_time = (time.time() - start_time) / 60
    print(f"\nALL BENCHMARKS COMPLETE. Total Suite Execution Time: {total_time:.2f} minutes.")

if __name__ == "__main__":
    main()