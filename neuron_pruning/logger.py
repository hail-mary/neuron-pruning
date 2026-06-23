import os
import yaml
import pprint
import cloudpickle
import warnings
import json
import numpy as np

class Logger:
    def __init__(self, cfg, save_cfg=True):     
        self.iteration = -1
        self.history = {
            "iterations": [],
            "mean_rewards": [],
            "std_rewards": [],
            "inference_flops": [],
            "iteration_training_flops": []
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

    def log_training_summary(self, start_time, end_time, all_rewards, best_policy_arch, best_iteration, **kwargs):
        # Calculate the duration
        duration_seconds = int(end_time - start_time)
        h, rem = divmod(duration_seconds, 3600)
        m, s = divmod(rem, 60)
        duration_str = f"{h:02d}:{m:02d}:{s:02d}"

        # Calculate Final score (mean over the last 10 evaluation checkpoints together with the std)
        last_10_rewards = all_rewards[-10:] if len(all_rewards) >= 10 else all_rewards
        final_score_mean = np.mean(last_10_rewards)
        final_score_std = np.std(last_10_rewards)
        
        best_reward = max(all_rewards) if all_rewards else 0

        # Create the summary
        summary = (
            f"Training Summary:\n"
            f"-----------------\n"
            f"Total Training Time: {duration_str}\n"
            f"Environment: {self.cfg.get('env_name', 'Unknown')}\n"
            f"Algorithm: {self.cfg.get('algorithm', 'Unknown')}\n"
            f"Target Sparsity: {self.cfg['target_sparsity']}\n"
            f"Update Interval: {self.cfg['update_interval']}\n"
            f"Best Episode Return: {best_reward:.2f} (Iteration {best_iteration})\n"
            f"Final Score (Last 10 Checkpoints): {final_score_mean:.2f} +/- {final_score_std:.2f}\n"
            f"Best Policy Architecture: {best_policy_arch}\n"
            f"Cumulative Training FLOPs: {kwargs.get('cumulative_training_flops', 0):.2e}\n"
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

    def log_iteration(self, mean_reward, std_reward, inference_flops, iteration_training_flops=0):
        # Log the data for each iteration
        self.history["iterations"].append(self.iteration)
        self.history["mean_rewards"].append(mean_reward)
        self.history["std_rewards"].append(std_reward)
        self.history["inference_flops"].append(inference_flops)
        self.history["iteration_training_flops"].append(iteration_training_flops)

    def save_history(self):
        # Save the history to a JSON file
        history_path = os.path.join(self.cfg['logdir'], 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)