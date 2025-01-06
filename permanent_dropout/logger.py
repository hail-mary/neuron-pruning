import os
import yaml
import pprint
import cloudpickle
import warnings
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


class Logger:
    def __init__(self, cfg):     
        self.iteration = 0
        # make log directory if not exists
        if not os.path.exists(cfg['logdir']):
            os.mkdir(cfg['logdir'])
        else:
            warnings.warn(f'{cfg["logdir"]} already exists. Logs may be overwritten.')

         # Pretty-print the configuration
        print('\n------------------ Loaded Configuration --------------------')
        pprint.pprint(cfg)

        # save configuration
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
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        else:
            warnings.warn(f'{checkpoints_dir} already exists. Checkpoints may be overwritten.')
        path_to_model = os.path.join(checkpoints_dir, f'Iteration-{idx}')
        
        if os.path.exists(path_to_model):
            warnings.warn(f'{path_to_model} already exists. Files may be overwritten.')
        else:
            os.mkdir(path_to_model)
        
        with open(f'{path_to_model}/best_policy_kwargs.pkl', 'wb') as f:
            cloudpickle.dump(best_policy_kwargs, f)
        with open(f'{path_to_model}/best_policy_weights.pkl', 'wb') as f:
            cloudpickle.dump(best_policy_weight, f)

    def plot_learning_curve(self, all_rewards, rewards_per_worker, neuron_counts):
        # Plot the learning curve with dual y-axes
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot all_rewards on the first y-axis
        ax1.set_xlabel('Environment steps [1e3]', fontsize=20)
        ax1.set_ylabel('Cumulative Reward', fontsize=20)
        ax1.plot(all_rewards, label='Mean Reward')
        ax1.tick_params(axis='y', labelsize=12)

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
        ax1.tick_params(axis='x', labelsize=12)

        # Create a second y-axis to plot total_neurons
        ax2 = ax1.twinx()
        ax2.set_ylabel('Total Neurons', fontsize=20)
        ax2.plot(neuron_counts, label='Total Neurons', color='red', linestyle='--')
        ax2.tick_params(axis='y', labelsize=12)

        # Ensure y-axis ticks for total neurons are integers
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        fig.tight_layout()
        plt.grid(True)

        # Add legends and bring them to the front
        legend1 = ax1.legend(loc='upper left', fontsize=14)
        legend2 = ax2.legend(loc='upper right', fontsize=14)
        legend1.set_zorder(10)
        legend2.set_zorder(10)

        # Show the plot
        plt.show()

    def log_training_summary(self, start_time, end_time, best_reward, average_reward, best_policy_arch):
        # Calculate the duration in hours
        duration_seconds = end_time - start_time
        duration_hours = duration_seconds / 3600

        # Create the summary
        summary = (
            f"Training Summary:\n"
            f"-----------------\n"
            f"Total Training Time: {duration_hours:.2f} hours\n"
            f"Best Reward: {best_reward}\n"
            f"Average Reward: {average_reward:.2f}\n"
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

       