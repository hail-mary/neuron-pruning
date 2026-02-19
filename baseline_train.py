import yaml
import argparse
import numpy as np
import torch
import time
import os
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from neuron_pruning.model import Model
from neuron_pruning.scheduler import Scheduler
from neuron_pruning.logger import Logger
from torch.utils.tensorboard import SummaryWriter

def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def calc_params(arch, cfg):
    obs_dim = cfg['obs_dim']
    action_dim = cfg['action_dim']
    arch_pi = [obs_dim] + arch['pi'] + [action_dim]
    arch_vf = [obs_dim] + arch['vf'] + [action_dim]
    params_pi = 0
    params_vf = 0
    for i in range(len(arch_pi) - 1):
        params_pi += arch_pi[i] * arch_pi[i+1]
    for i in range(len(arch_vf) - 1):
        params_vf += arch_vf[i] * arch_vf[i+1]
    
    return params_pi, params_vf

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train the baseline pruned model.')
    parser.add_argument('--cfg', type=str, default='config.yaml', help='Path to the configuration file.')
    parser.add_argument('--env', type=str, help='Specify the environment name to override the config file.')
    parser.add_argument('--logdir', type=str, help='Specify the directory for logging.')
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.cfg)

    # Override config if specified
    if args.env:
        cfg['env_name'] = args.env
    if args.logdir:
        cfg['logdir'] = args.logdir

    # 1. Initialize environment and dimensions
    temp_env = gym.make(cfg['env_name'])
    cfg['obs_dim'] = temp_env.observation_space.shape[0]
    cfg['action_dim'] = temp_env.action_space.shape[0]
    temp_env.close()

    # Initialize Scheduler and Logger
    scheduler = Scheduler(cfg)
    logger = Logger(cfg)

    # 2. Setup Multi-processed Environment and Training Model
    num_workers = cfg['num_workers']
    train_env = make_vec_env(cfg['env_name'], n_envs=num_workers, vec_env_cls=SubprocVecEnv, seed=cfg['seed'])
    
    # Initialize model with dense architecture
    model = Model(cfg)
    # Ensure policy is created with the vectorized environment
    model.make_policy(train_env)

    # Track current architecture and FLOPs
    current_arch = cfg['policy_kwargs']['net_arch'].copy()
    params_pi, params_vf = calc_params(current_arch, cfg)
    total_flops = (params_pi + params_vf) * 2
    
    print("\nInitial Policy Summary:")
    for layer, shape in current_arch.items():
        flops = params_pi * 2 if layer == 'pi' else params_vf * 2
        print(f"  - {layer}: {shape} FLOPs: {flops}")
    print(f"Total FLOPs: {total_flops}")

    # 4. Training Loop
    writer = SummaryWriter(cfg['logdir'])
    print('\n#---------------------- Start Baseline Training ! -----------------------#')
    start_time = time.time()
    
    all_rewards = []
    flops_counts = []

    for iteration in range(cfg['num_iterations']):
        logger.step()
        
        # Train: learn() handles data collection from all workers and update
        model.learn(total_timesteps=num_workers * cfg['timesteps_per_iteration'])
        
        # Periodic Evaluation
        mean_reward, std_reward = model.evaluate_policy(seed=cfg['seed'], num_eval_episodes=10)
        
        all_rewards.append(mean_reward)
        flops_counts.append(total_flops)
        
        writer.add_scalar('eval/avg_return', mean_reward, iteration)
        writer.add_scalar('eval/flops', total_flops, iteration)
        
        # Log statistics
        logger.log_iteration(mean_reward, 0.0, total_flops)
        
        print(f"Iteration {iteration}/{cfg['num_iterations']}: Reward = {mean_reward:.2f}, FLOPs = {total_flops}")

        # --- Pruning Trigger (Same logic as main.py) ---
        if iteration > 0 and iteration % cfg['update_interval'] == 0:
            print(f"\n----------- Iteration {iteration}/{cfg['num_iterations']}: Modifying network architecture ----------")
            
            # Get current model weights
            raw_params = model.policy.state_dict()
            
            # Apply pruning
            arch_input, params_input, aux = scheduler.preprocess(raw_arch=current_arch, raw_params=raw_params)
            modified_arch, modified_params = scheduler.modify_network(params_input, arch_input, iteration, cfg['target_sparsity'])
            current_arch, next_params = scheduler.reconstruct(modified_arch, modified_params, aux)
            
            # Update FLOPs count
            params_pi, params_vf = calc_params(current_arch, cfg)
            total_flops = (params_pi + params_vf) * 2
            
            # Reconstruct the policy with the new architecture and weights
            new_policy_kwargs = cfg['policy_kwargs'].copy()
            new_policy_kwargs['net_arch'] = current_arch
            model.make_policy(train_env, policy_kwargs=new_policy_kwargs, policy_weights=next_params)
            
            print(f"New Architecture: {current_arch}, New FLOPs: {total_flops}")

    end_time = time.time()

    # 5. Finalizing and Logging
    # Plotting and summary
    best_reward = max(all_rewards)
    best_iteration = np.argmax(all_rewards)
    average_reward = np.mean(all_rewards)

    logger.save_history()
    # rewards_per_worker is expected for range plotting, we use the same list as there's only one model
    logger.plot_learning_curve(all_rewards, [all_rewards] * num_workers, flops_counts)
    
    logger.log_training_summary(
        start_time=start_time, 
        end_time=end_time, 
        best_reward=best_reward, 
        average_reward=average_reward, 
        best_policy_arch=current_arch, 
        best_iteration=best_iteration
    )

    print(f"\nTraining Complete. Models and logs are saved in {cfg['logdir']}")

if __name__ == "__main__":
    main()
