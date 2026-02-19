import os
import yaml
import pprint
import cloudpickle
import warnings
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import json
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["font.size"] = 22
plt.rcParams['xtick.labelsize'] = 20 
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['lines.linewidth'] = 3.0
plt.rcParams["legend.framealpha"] = 1

class Logger:
    def __init__(self, cfg, save_cfg=True):     
        self.iteration = -1
        self.history = {
            "iterations": [],
            "mean_rewards": [],
            "std_rewards": [],
            "flops_counts": []
        }
        # make log directory if not exists
        if not os.path.exists(cfg['logdir']):
            os.makedirs(cfg['logdir'])
        else:
            warnings.warn(f'{cfg["logdir"]} already exists. Logs may be overwritten.')

        # save configuration
        if save_cfg:
            # Pretty-print the configuration
            print('\n#------------------ Loaded Configuration --------------------#')
            pprint.pprint(cfg)
            config_dir = os.path.join(cfg['logdir'], 'config.yaml')
            with open(config_dir, 'w') as file:
                yaml.dump(cfg, file, indent=4)
                print(f'\n >> saved to {config_dir}.')

        self.cfg = cfg
    
    def step(self):
        self.iteration += 1
    
    def save_checkpoint(self, results, save_to='checkpoints'):
        # results = [[worker_id, reward, policy_kwargs, policy_weight],
        #              worker_id, reward, policy_kwargs, policy_weight], ... ]

        # sort results in reward decending order
        results.sort(key=lambda x: x[1], reverse=True)

        # find the best worker and save at every checkpoints
        best_worker = results[0][0]
        best_reward = results[best_worker][1]
        best_policy_kwargs = results[best_worker][2]
        best_policy_weight = results[best_worker][3]

        idx = self.iteration
        checkpoints_dir = os.path.join(self.cfg['logdir'], save_to)
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        path_to_model = os.path.join(checkpoints_dir, f'Iteration-{idx}')
        if os.path.exists(path_to_model):
            warnings.warn(f'{path_to_model} already exists. Files may be overwritten.')
        
        os.makedirs(path_to_model, exist_ok=True)
        
        with open(f'{path_to_model}/best_policy_kwargs.pkl', 'wb') as f:
            cloudpickle.dump(best_policy_kwargs, f)
        with open(f'{path_to_model}/best_policy_weights.pkl', 'wb') as f:
            cloudpickle.dump(best_policy_weight, f)

    def plot_learning_curve(self, all_rewards, rewards_per_worker, flops_counts):
        # Plot the learning curve with dual y-axes
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot all_rewards on the first y-axis
        ax1.set_xlabel('Environment steps [1e3]', fontsize=20)
        ax1.set_ylabel('Cumulative Reward', fontsize=20)
        ax1.plot(all_rewards, label='Mean Reward')
        ax1.tick_params(axis='y', labelsize=15)

        # Fill between for mean reward range
        iterations = range(len(all_rewards))
        rewards_min = np.min(rewards_per_worker, axis=0)
        rewards_max = np.max(rewards_per_worker, axis=0)
        ax1.fill_between(
            iterations,
            rewards_min,
            rewards_max,
            color="blue",
            alpha=0.2,
            label="Reward Range",
        )

        # Ensure x-axis ticks are integers
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax1.tick_params(axis='x', labelsize=15)
        ax1.set_xticks(np.arange(0, self.cfg['num_iterations']+1, step=100))  # Adjust step as needed

        # Create a second y-axis to plot total_flops
        ax2 = ax1.twinx()
        ax2.set_ylabel('Total FLOPs', fontsize=20)
        ax2.plot(flops_counts, label='Total FLOPs', color='red', linestyle='--')
        ax2.tick_params(axis='y', labelsize=15)

        # Ensure y-axis ticks for total FLOPs are integers
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        fig.tight_layout()
        plt.grid(True)

        # Add legends and bring them to the front
        legend = ax1.legend(loc='lower left', frameon=True, framealpha=1.0)
        legend.set_zorder(100)  # Set a higher zorder to ensure it's on top
        plt.draw()  # Force a redraw to ensure legend is on top

        # Determine the maximum iteration and format it
        max_iteration = len(all_rewards) * self.cfg['timesteps_per_iteration']  # Assuming each step represents 1000 iterations
        if max_iteration >= 1_000_000:
            iteration_str = f"{max_iteration // 1_000_000}M"
        elif max_iteration >= 1_000:
            iteration_str = f"{max_iteration // 1_000}K"
        else:
            iteration_str = str(max_iteration)

        # Save the plot as a PNG file with the formatted iteration count
        png_file_path = os.path.join(self.cfg['logdir'], f'learning_curve_{iteration_str}.png')
        plt.savefig(png_file_path, format='png')
        print(f"Learning curve saved as {png_file_path}")

        # Save the training history
        self.save_history()

    def log_training_summary(self, start_time, end_time, best_reward, average_reward, best_policy_arch, best_iteration):
        # Calculate the duration in hours
        duration_seconds = end_time - start_time
        duration_hours = duration_seconds / 3600

        # Create the summary
        summary = (
            f"Training Summary:\n"
            f"-----------------\n"
            f"Total Training Time: {duration_hours:.2f} hours\n"
            f"Target Sparsity: {self.cfg['target_sparsity']}\n"
            f"Update Interval: {self.cfg['update_interval']}\n"
            f"Best Reward: {best_reward}\n"
            f"Best Iteration: {best_iteration}\n"
            f"Best Policy Architecture: {best_policy_arch}\n"
        )

        # Print the summary
        print(summary)

        # Construct the directory path from the configuration
        logdir = self.cfg['logdir']
        os.makedirs(logdir, exist_ok=True)

        # Log the summary to a file in the specified directory
        summary_path = os.path.join(logdir, 'training_summary.txt')
        with open(summary_path, 'a') as summary_file:
            summary_file.write(summary)

    def log_iteration(self, mean_reward, std_reward, total_flops):
        # Log the data for each iteration
        self.history["iterations"].append(self.iteration)
        self.history["mean_rewards"].append(mean_reward)
        self.history["std_rewards"].append(std_reward)
        self.history["flops_counts"].append(total_flops)

    def save_history(self):
        # Save the history to a JSON file
        history_path = os.path.join(self.cfg['logdir'], 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def plot_learning_curve_from_json(self, json_files, plot1_every=100, plot2_every=10):
        # Initialize the plot
        fig, ax1 = plt.subplots(figsize=(10, 5))
        labels = ['OrdRL+WA', 'NRM+WA', 'Proposed']
        markers = ['s', '^', 'o']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        markersize = 9

        # Iterate over each JSON file
        for i, json_file in enumerate(json_files):
            with open(json_file, 'r') as f:
                history = json.load(f)

            # Select every 10th iteration
            iterations = history['iterations'][::plot1_every] + [history['iterations'][-1]]
            mean_rewards = history['mean_rewards'][::plot1_every] + [history['mean_rewards'][-1]]
            std_rewards = history['std_rewards'][::plot1_every] + [history['std_rewards'][-1]]
            iterations2 = history['iterations'][::plot2_every]
            flops_counts = history['flops_counts'][::plot2_every]

            # Plot mean_rewards on the first y-axis
            ax1.plot(iterations, mean_rewards, label=labels[i], marker=markers[i], markersize=markersize, color=colors[i])
            ax1.fill_between(iterations, 
                             np.array(mean_rewards) - np.array(std_rewards), 
                             np.array(mean_rewards) + np.array(std_rewards), 
                             alpha=0.2, color=colors[i])
           
        # Set labels and ticks for the first y-axis
        ax1.set_xlabel(r'Environment steps [$\times$ 2048]')
        ax1.set_ylabel('Episode Return')
        ax1.tick_params(axis='y')
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax1.tick_params(axis='x')
        ax1.set_xticks(np.arange(0, self.cfg['num_iterations']+1, step=100))  # Adjust step as needed
        # ax1.grid()
        # Create a second y-axis to plot total_flops
        ax2 = ax1.twinx()
        for j, json_file in enumerate(json_files):
            with open(json_file, 'r') as f:
                history = json.load(f)
            flops_counts = history['flops_counts'][::plot2_every]
            ax2.plot(iterations2, flops_counts, linestyle='--', marker=markers[j], markevery=10, markersize=markersize, color=colors[j])

        # Set labels and ticks for the second y-axis
        ax2.set_ylabel('Total FLOPs')
        ax2.tick_params(axis='y')
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        fig.tight_layout()
        ax2.grid(True, zorder=0, axis='y')
        # Add legends and bring them to the front
        legend = ax1.legend(loc='lower left', frameon=True, framealpha=1.0)
        # legend.set_zorder(10)  # Set a higher zorder to ensure it's on top
        plt.show()