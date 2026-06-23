import subprocess
import os
import time
import sys

def run_command(cmd, name):
    print(f"\n>>> Running {name}...")
    print(f"Command: {' '.join(cmd)}")
    start = time.time()
    try:
        # Use subprocess.Popen or run without capture to see output
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        print(f"DONE in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {e}")
        return False

def main():
    # Test parameters: short runs to verify no crashes and logic execution
    env = "Pendulum-v1" # Continuous control, fast
    iters = 2
    update_interval = 1
    
    # 1. Proposed (main.py)
    cmd_proposed = [
        sys.executable, "main.py",
        "--env", env,
        "--num_iterations", str(iters),
        "--update_interval", str(update_interval),
        "--logdir", "test_logs/proposed",
        "--seed", "42"
    ]
    run_command(cmd_proposed, "Proposed (Pruning + Merge + Reset)")

    # 2. PPO-WA (main.py with high update_interval)
    cmd_ppo_wa = [
        sys.executable, "main.py",
        "--env", env,
        "--num_iterations", str(iters),
        "--update_interval", "100", # No pruning
        "--logdir", "test_logs/ppo_wa",
        "--seed", "42"
    ]
    run_command(cmd_ppo_wa, "PPO-WA (Merge only)")

    # 3. Structured-GMP (baseline_train.py)
    cmd_gmp = [
        sys.executable, "baseline_train.py",
        "--env", env,
        "--num_iterations", str(iters),
        "--update_interval", str(update_interval),
        "--logdir", "test_logs/gmp"
    ]
    run_command(cmd_gmp, "Structured-GMP (Pruning only + No Reset)")

if __name__ == "__main__":
    main()
