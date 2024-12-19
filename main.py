import yaml
import json
import gymnasium as gym
import numpy as np
from multiprocessing import Process, Queue
import stable_baselines3
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
from permanent_dropout.model import Model
from permanent_dropout.trainer import Trainer


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def follower_process(queue, weight_queue, worker_id, cfg):
    """
    各workerプロセスでPPOを学習し、評価結果をleaderに送信。
    """
    # Create variable architecture model
    model = Model(cfg)
    env = model.env

    for iteration in range(cfg['num_iterations']):
        # Update policy weights
        model.learn(total_timesteps=cfg['timesteps_per_iteration'])

        # policy evaluation
        total_reward = model.evaluate_policy()

        # send current policy weights and architecture to the leader
        queue.put((worker_id, iteration, total_reward, model.policy_kwargs, model.policy.state_dict()))

        # receive new policy weights and architecture from the leader
        new_arch, new_weights = weight_queue.get()
        new_kwargs = model.policy_kwargs.copy()
        new_kwargs['net_arch'] = new_arch
        new_model = model.make_policy(env, new_kwargs, new_weights)
        model.model = new_model

    # Save the final model
    final_model_path = os.path.join(cfg['logdir'], 'terminal')
    model.save_policy(save_to=final_model_path)

def leader_process(queue, weight_queue, cfg):
    """
    leaderが評価結果を収集し、収益を平滑化、傾きを計算、必要ならネットワークを再構築。
    """
    num_workers = cfg['num_workers']
    num_iterations = cfg['num_iterations']
    ema_window = cfg['ema_window']
    dropout_rates = cfg['dropout_rates']
    slope_threshold = cfg['slope_threshold']
    device = cfg['device']
    trainer = Trainer(cfg)

    all_rewards = []
    ema_rewards = []
    rewards_per_worker = [[] for _ in range(num_workers)]  # 各workerの収益記録
    neuron_counts = []  # ニューロン数の推移
    iteration_list = []

    for iteration in range(num_iterations):
        rewards = []
        weights = []
        policy_arch = None
        iteration_list.append(iteration)
        results = []

        # 各workerから結果を収集
        for _ in range(num_workers):
            worker_id, iter_id, reward, policy_kwargs, policy_weight = queue.get()
            rewards.append(reward)
            rewards_per_worker[worker_id].append(reward)
            weights.append(policy_weight)
            policy_arch = policy_kwargs["net_arch"]
            results.append((worker_id, reward, policy_kwargs, policy_weight))

        # sort results in reward decending order
        results.sort(key=lambda x: x[1], reverse=True)

        # find the best worker and save at every checkpoint
        if iteration > 0 and iteration % cfg['checkpoint_interval'] == 0:
            best_worker = results[0][0]
            best_policy_kwargs = results[best_worker][2]
            best_policy_weight = results[best_worker][3]
            best_actor = best_policy_kwargs['net_arch']['pi']
            best_critic = best_policy_kwargs['net_arch']['vf']
            print(best_worker)
            

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
        if slope < slope_threshold and len(weights) > 0:
            print(f"-----------Iteration {iteration}: Modifying network architecture----------")
            avg_weights = trainer.average_weights(weights)
            modified_arch, modified_weights = trainer.modify_network(avg_weights, policy_arch, dropout_rates, device)
            iteration_list = []
            ema_rewards = []
        else:
            modified_arch = policy_arch
            modified_weights = trainer.average_weights(weights)

        # ニューロン数を記録
        total_neurons = sum(sum(units) for units in modified_arch.values())
        neuron_counts.append(total_neurons)

        # 重みと新しいアーキテクチャを各workerに送信
        for _ in range(num_workers):
            weight_queue.put((modified_arch, modified_weights))

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


if __name__ == "__main__":
    import pprint
    cfg = load_config('config.yaml')
    # Pretty-print the configuration
    pprint.pprint(cfg)

    result_queue = Queue()
    weight_queue = Queue()

    # Workerプロセスの作成
    workers = []
    for worker_id in range(cfg['num_workers']):
        p = Process(target=follower_process, args=(result_queue, weight_queue, worker_id, cfg))
        workers.append(p)
        p.start()

    # Leaderプロセスの実行
    leader_process(result_queue, weight_queue, cfg)

    # Workerプロセスの終了を待つ
    for p in workers:
        p.join()
