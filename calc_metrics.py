import glob
import numpy as np

# Specify the environment and method to analyze
# env: "Ant-v5", "HalfCheetah-v5", "Swimmer-v5", "Walker2d-v5"
# method: "proposed", "PPO-WA", "Structured-GMP"
env = "Walker2d-v5"
method = "proposed"

# Search in both data and data_eval_fix directories
file_patterns = [
    f"data/{env}/{method}/seed-*/training_summary.txt",
    f"data_eval_fix/{env}/{method}/seed-*/training_summary.txt"
]

files = []
for pattern in file_patterns:
    files.extend(glob.glob(pattern))

# Lists to store the extracted values
total_training_times = []
best_rewards = []

# Iterate over each file and extract the required information
for file_path in files:
    with open(file_path, 'r') as file:
        total_training_time = None
        best_reward = None
        for line in file:
            if "Total Training Time" in line:
                # Parse "Total Training Time: 01:08:14" or "0.5 hours"
                time_str = line.split(":")[-1].strip()
                if "hours" in time_str:
                    total_training_time = float(time_str.replace(" hours", ""))
                else:
                    # HH:MM:SS format
                    parts = line.replace("Total Training Time:", "").strip().split(":")
                    if len(parts) == 3:
                        total_training_time = int(parts[0]) + int(parts[1]) / 60.0 + int(parts[2]) / 3600.0
            elif "Best Episode Return" in line or "Best Reward" in line:
                # Parse "Best Episode Return: 4416.01 (Iteration 892)" or "Best Reward: 4416.01"
                val_part = line.split(":")[-1].strip()
                best_reward = float(val_part.split()[0])
        
        # Store the extracted information
        if total_training_time is not None and best_reward is not None:
            total_training_times.append(total_training_time)
            best_rewards.append(best_reward)

# Calculate and print mean and standard deviation
if total_training_times and best_rewards:
    mean_training_time = np.mean(total_training_times)
    std_training_time = np.std(total_training_times, ddof=1) if len(total_training_times) > 1 else 0.0
    mean_best_reward = np.mean(best_rewards)
    std_best_reward = np.std(best_rewards, ddof=1) if len(best_rewards) > 1 else 0.0

    print(f"Method: {method}")
    print(f"Env: {env}")
    print(f"Data count: {len(total_training_times)}")
    print(f"Best Rewards: {best_rewards}")
    print(f"Mean Total Training Time: {mean_training_time:.2f} hours")
    print(f"Standard Deviation of Total Training Time: {std_training_time:.2f} hours")
    print(f"Mean Best Reward: {mean_best_reward:.2f}")
    print(f"Standard Deviation of Best Reward: {std_best_reward:.2f}")
else:
    print(f"No valid data found for Env: {env}, Method: {method}")