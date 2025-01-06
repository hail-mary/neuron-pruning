import os
import yaml
import argparse
import pprint
import numpy as np
from multiprocessing import Process, Queue
import time

from permanent_dropout.model import Model
from permanent_dropout.trainer import Trainer
from permanent_dropout.logger import Logger


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def eval_only(cfg, load_from):
    # Load the model
    model = Model(cfg)
    model.load_policy(load_from)  # Assuming you have a method to load the trained policy

    # Evaluate the policy
    print('\n------------------- Start Evaluation ! ---------------------')
    total_reward = model.evaluate_policy()
    print(model.policy_kwargs["net_arch"])
    print(f"Evaluation Reward: {total_reward}")

def follower_process(result_queue, network_queue, worker_id, cfg):
    """
    Each worker process learns and sends evaluation results to the leader.
    """
    # Create variable architecture model
    model = Model(cfg)
    env = model.env

    for iteration in range(cfg['num_iterations']):
        # Update policy params
        model.learn(total_timesteps=cfg['timesteps_per_iteration'])

        # Policy evaluation
        total_reward = model.evaluate_policy()

        # Send current policy params and architecture to the leader
        result_queue.put((worker_id, total_reward, model.policy_kwargs, model.policy.state_dict()))

        # Receive new policy params and architecture from the leader
        new_arch, new_params = network_queue.get()

        if model.policy_kwargs['net_arch'] != new_arch:  # Net architecture has been changed
            new_kwargs = model.policy_kwargs.copy()
            new_kwargs['net_arch'] = new_arch
            model.make_policy(env, new_kwargs, new_params)
        else:
            model.make_policy(env, model.policy_kwargs, new_params)

def leader_process(result_queue, network_queue, cfg):
    """
    The leader collects evaluation results, smooths rewards, calculates slope, and reconstructs the network if necessary.
    """
    # Start timing the training
    start_time = time.time()

    num_workers = cfg['num_workers']
    num_iterations = cfg['num_iterations']
    ema_window = cfg['ema_window']
    dropout_rates = cfg['dropout_rates']
    slope_threshold = cfg['slope_threshold']
    trainer = Trainer(cfg)
    logger = Logger(cfg)

    all_rewards = []
    ema_rewards = []
    rewards_per_worker = [[] for _ in range(num_workers)]  # Record rewards for each worker
    neuron_counts = []  # Track neuron count changes
    iteration_list = []
    best_policy_arch = None
    best_reward = float('-inf')

    for iteration in range(num_iterations):
        terminate = iteration == num_iterations - 1
        if iteration == 0:
            print('\n---------------------- Start Training ! -----------------------')
        rewards = []
        params = []
        policy_arch = None
        iteration_list.append(iteration)
        results = []
        logger.step()

        # Collect results from each worker
        for _ in range(num_workers):
            worker_id, reward, policy_kwargs, policy_weight = result_queue.get()
            rewards.append(reward)
            rewards_per_worker[worker_id].append(reward)
            params.append(policy_weight)
            policy_arch = policy_kwargs["net_arch"]
            results.append((worker_id, reward, policy_kwargs, policy_weight))

            # Track best reward and corresponding architecture
            if reward > best_reward:
                best_reward = reward
                best_policy_arch = policy_arch.copy()

        if iteration > 0 and iteration % cfg['checkpoint_interval'] == 0 or terminate:
            logger.save_checkpoint(results)

        # Display statistics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        all_rewards.append(mean_reward)

        # Update EMA
        alpha = 2 / (ema_window + 1)
        if ema_rewards:
            ema_reward = alpha * mean_reward + (1 - alpha) * ema_rewards[-1]
        else:
            ema_reward = mean_reward
        ema_rewards.append(ema_reward)

        # Calculate slope
        if len(ema_rewards) > ema_window:
            slope = (np.polyfit(iteration_list, ema_rewards, deg=1))[0]
        else:
            slope = float("nan")

        # Output: statistics
        print(f"\nIteration {iteration}: Reward Mean = {mean_reward:.2f}, Std = {std_reward:.2f}, Slope = {slope:.4f}")

        # Output: architecture information
        print("Current Policy Architecture:")
        for layer, shape in policy_arch.items():
            print(f"  - {layer}: {shape}")
        
        # Network modification condition
        if slope < slope_threshold and len(params) > 0:
            print(f"\n----------- Iteration {iteration}: Modifying network architecture ----------")
            if cfg['average_weights']:
                avg_params = trainer.average_params(params)
                arch, params, aux = trainer.preprocess(raw_arch=policy_arch, raw_params=avg_params)
                modified_arch, modified_params = trainer.modify_network(params, arch, dropout_rates)
                modified_arch, modified_params = trainer.reconstruct(modified_arch, modified_params, aux)
            else:
                # If not averaging weights, modify each worker's params individually
                for worker_id in range(num_workers):
                    arch, params[worker_id], aux = trainer.preprocess(raw_arch=policy_arch.copy(), raw_params=params[worker_id])
                    modified_arch, modified_params[worker_id] = trainer.modify_network(params[worker_id], arch, dropout_rates)
                    modified_arch, modified_params[worker_id] = trainer.reconstruct(modified_arch, modified_params[worker_id], aux)
            
            iteration_list = []
            ema_rewards = []

        else:
            modified_arch = policy_arch

            if cfg['average_weights']:
                modified_params = trainer.average_params(params)
            else:
                # Set different params for each worker
                modified_params = [params[worker_id] for worker_id in range(num_workers)]

        # Record neuron count
        total_neurons = sum(sum(units) for units in modified_arch.values())
        neuron_counts.append(total_neurons)

        # Send weights and new architecture to each worker
        for worker_id in range(num_workers):
            if cfg['average_weights']:
                network_queue.put((modified_arch, modified_params))
            else:
                network_queue.put((modified_arch, modified_params[worker_id]))   

    logger.plot_learning_curve(all_rewards, rewards_per_worker, neuron_counts)

    # End timing the training
    end_time = time.time()

    # Calculate average reward
    average_reward = sum(all_rewards) / len(all_rewards)

    # Log the training summary using the Logger class
    logger.log_training_summary(start_time, end_time, best_reward, average_reward, best_policy_arch)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train or evaluate the policy.')
    parser.add_argument('--eval', type=str, help='Only evaluate a trained policy. Specify the directory to load the policy from.')
    parser.add_argument('--logdir', type=str, help='Specify the directory for logging.')
    args = parser.parse_args()

    # Load configuration
    cfg = load_config('config.yaml')
    
    if args.logdir:
        cfg['logdir'] = args.logdir

    # Assert if dropout rates and net_arch are consistent
    assert len(cfg['dropout_rates']['policy']) == len(cfg['policy_kwargs']['net_arch']['pi'])
    assert len(cfg['dropout_rates']['value']) == len(cfg['policy_kwargs']['net_arch']['vf'])

    if args.eval:
        # Evaluate the policy
        eval_only(cfg, load_from=args.eval)
    else:
        # Training
        result_queue = Queue()
        network_queue = Queue()

        # Worker processes
        workers = []
        for worker_id in range(cfg['num_workers']):
            p = Process(target=follower_process, args=(result_queue, network_queue, worker_id, cfg))
            workers.append(p)
            p.start()

        # Leader process
        leader_process(result_queue, network_queue, cfg)

        # Wait for worker processes to finish
        for p in workers:
            p.join()

if __name__ == "__main__":
    main()
