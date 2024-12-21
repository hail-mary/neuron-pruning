import os
import yaml
import argparse
import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from multiprocessing import Process, Queue

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
    各workerプロセスでPPOを学習し、評価結果をleaderに送信。
    """
    # Create variable architecture model
    model = Model(cfg)
    env = model.env

    for iteration in range(cfg['num_iterations']):
        # Update policy params
        model.learn(total_timesteps=cfg['timesteps_per_iteration'])

        # policy evaluation
        total_reward = model.evaluate_policy()

        # send current policy params and architecture to the leader
        result_queue.put((worker_id, total_reward, model.policy_kwargs, model.policy.state_dict()))

        # receive new policy params and architecture from the leader
        new_arch, new_params = network_queue.get()

        if model.policy_kwargs['net_arch'] != new_arch: # net arch has been changed
            new_kwargs = model.policy_kwargs.copy()
            new_kwargs['net_arch'] = new_arch
            model.make_policy(env, new_kwargs, new_params)
        else:
            model.make_policy(env, model.policy_kwargs, new_params)

def leader_process(result_queue, network_queue, cfg):
    """
    leaderが評価結果を収集し、収益を平滑化、傾きを計算、必要ならネットワークを再構築。
    """
    num_workers = cfg['num_workers']
    num_iterations = cfg['num_iterations']
    ema_window = cfg['ema_window']
    dropout_rates = cfg['dropout_rates']
    slope_threshold = cfg['slope_threshold']
    trainer = Trainer(cfg)
    logger = Logger(cfg)

    all_rewards = []
    ema_rewards = []
    rewards_per_worker = [[] for _ in range(num_workers)]  # 各workerの収益記録
    neuron_counts = []  # ニューロン数の推移
    iteration_list = []

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

        # 各workerから結果を収集
        for _ in range(num_workers):
            worker_id, reward, policy_kwargs, policy_weight = result_queue.get()
            rewards.append(reward)
            rewards_per_worker[worker_id].append(reward)
            params.append(policy_weight)
            policy_arch = policy_kwargs["net_arch"]
            results.append((worker_id, reward, policy_kwargs, policy_weight))

        if iteration > 0 and iteration % cfg['checkpoint_interval'] == 0 or terminate:
            logger.save_checkpoint(results)
            

        # 統計情報を表示
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        all_rewards.append(mean_reward)

        # EMAを更新
        alpha = 2 / (ema_window + 1)
        if ema_rewards:
            ema_reward = alpha * mean_reward + (1 - alpha) * ema_rewards[-1]
        else:
            ema_reward = mean_reward
        ema_rewards.append(ema_reward)

        # 傾きを計算
        if len(ema_rewards) > ema_window:
            slope = (np.polyfit(iteration_list, ema_rewards, deg=1))[0]
        else:
            slope = float("nan")

        # 出力: 統計情報
        print(f"\nIteration {iteration}: Reward Mean = {mean_reward:.2f}, Std = {std_reward:.2f}, Slope = {slope:.4f}")

        # 出力: アーキテクチャ情報
        print("Current Policy Architecture:")
        for layer, shape in policy_arch.items():
            print(f"  - {layer}: {shape}")

        # ネットワークの修正条件
        if slope < slope_threshold and len(params) > 0:
            print(f"\n----------- Iteration {iteration}: Modifying network architecture ----------")
            avg_params = trainer.average_params(params)
            arch, params, aux = trainer.preprocess(raw_arch=policy_arch, raw_params=avg_params)
            modified_arch, modified_params = trainer.modify_network(params, arch, dropout_rates)
            modified_arch, modified_params = trainer.reconstruct(modified_arch, modified_params, aux)

            iteration_list = []
            ema_rewards = []
        else:
            modified_arch = policy_arch
            modified_params = trainer.average_params(params)

        # ニューロン数を記録
        total_neurons = sum(sum(units) for units in modified_arch.values())
        neuron_counts.append(total_neurons)

        # 重みと新しいアーキテクチャを各workerに送信
        for _ in range(num_workers):
            network_queue.put((modified_arch, modified_params))

    # Plot the learning curve with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot all_rewards on the first y-axis
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Reward', color='tab:blue')
    ax1.plot(all_rewards, label='Mean Reward', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

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

    # Create a second y-axis to plot total_neurons
    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Neurons', color='tab:orange')
    ax2.plot(neuron_counts, label='Total Neurons', color='tab:orange', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Ensure y-axis ticks for total neurons are integers
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Add a title and grid
    plt.title('Learning Curve and Neuron Count')
    fig.tight_layout()
    plt.grid(True)

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.show()


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

    # assert if dropout rates and net_arch are consistent
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
