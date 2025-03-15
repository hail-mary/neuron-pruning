import glob
import statistics

# Specify the string that should be contained in the directory name
method = "nrm"
env = "walker"

# Use glob to find all matching files
file_pattern = f"./logs/{method}*{env}*/training_summary.txt"
files = glob.glob(file_pattern)

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
                total_training_time = float(line.split(":")[-1].strip().replace(" hours", ""))
            elif "Best Reward" in line:
                best_reward = float(line.split(":")[-1].strip())
        
        # Store the extracted information
        if total_training_time is not None and best_reward is not None:
            total_training_times.append(total_training_time)
            best_rewards.append(best_reward)

# Calculate and print mean and standard deviation
if total_training_times and best_rewards:
    mean_training_time = statistics.mean(total_training_times)
    std_training_time = statistics.stdev(total_training_times)
    mean_best_reward = statistics.mean(best_rewards)
    std_best_reward = statistics.stdev(best_rewards)

    print(f"Method: {method}")
    print(f"Env: {env}")
    print(f"Data: {len(total_training_times)}")
    print(f"Mean Total Training Time: {mean_training_time:.2f}")
    print(f"Standard Deviation of Total Training Time: {std_training_time:.2f}")
    print(f"Mean Best Reward: {mean_best_reward:.2f}")
    print(f"Standard Deviation of Best Reward: {std_best_reward:.2f}")