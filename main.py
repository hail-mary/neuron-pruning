import gymnasium as gym
import numpy as np
from multiprocessing import Process, Queue
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from .config import Config

os.environ["MUJOCO_GL"] = "egl"

def follower_process(env_name, queue, weight_queue, worker_id, num_iterations, device):
    """
    各workerプロセスでPPOを学習し、評価結果をleaderに送信。
    """
    env = gym.make(env_name, render_mode='rgb_array')
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, device=device)

    for iteration in range(num_iterations):
        model.learn(total_timesteps=1000)

        # 評価
        total_reward = evaluate_policy(env, model)

        # 現在のポリシー重みとアーキテクチャを送信
        queue.put((worker_id, iteration, total_reward, policy_kwargs, model.policy.state_dict()))

        # Leaderから新しい重みとアーキテクチャを受信
        new_weights, new_arch = weight_queue.get()
        model.policy.load_state_dict(new_weights)
        policy_kwargs["net_arch"] = new_arch

    env.close()

def leader_process(queue, weight_queue, num_workers, num_iterations, ema_window, dropout_rates, slope_threshold, device):
    """
    leaderが評価結果を収集し、収益を平滑化、傾きを計算、必要ならネットワークを再構築。
    """
    all_rewards = []
    ema_rewards = []
    rewards_per_worker = [[] for _ in range(num_workers)]  # 各workerの収益記録
    neuron_counts = []  # ニューロン数の推移
    iteration_list = []

    for iteration in range(num_iterations):
        results = []
        weights = []
        policy_arch = None
        iteration_list.append(iteration)

        # 各workerから結果を収集
        for _ in range(num_workers):
            worker_id, iter_id, reward, policy_kwargs, policy_weight = queue.get()
            results.append(reward)
            weights.append(policy_weight)
            rewards_per_worker[worker_id].append(reward)
            policy_arch = policy_kwargs["net_arch"]

        # 統計情報を表示
        mean_reward = np.mean(results)
        std_reward = np.std(results)
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
        print(f"Iteration {iteration}: Reward Mean = {mean_reward:.2f}, Std = {std_reward:.2f}, Slope = {slope:.4f}")

        # 出力: アーキテクチャ情報
        print("Current Policy Architecture:")
        for layer, shape in policy_arch.items():
            print(f"  - {layer}: {shape}")

        # ネットワークの修正条件
        if slope < slope_threshold and len(weights) > 0:
            print(f"Iteration {iteration}: Modifying network architecture")
            avg_weights = average_weights(weights)
            modified_arch, modified_weights = modify_network(avg_weights, policy_arch, dropout_rates, device)
            iteration_list = []
            ema_rewards = []
        else:
            modified_arch = policy_arch
            modified_weights = average_weights(weights)

        # ニューロン数を記録
        total_neurons = sum(sum(units) for units in modified_arch.values())
        neuron_counts.append(total_neurons)

        # 重みと新しいアーキテクチャを各workerに送信
        for _ in range(num_workers):
            weight_queue.put((modified_weights, modified_arch))

    # 収益のグラフを表示
    logger.log()


if __name__ == "__main__":
    ENV_NAME = "Ant-v4"
    NUM_WORKERS = 5
    NUM_ITERATIONS = 20
    EMA_WINDOW = 4  # 平滑化するIteration
    DROPOUT_RATES = {
    "pi": [0.05, 0.1],  # Actorネットワーク各層のドロップアウト率
    "vf": [0.0, 0.0]   # Criticネットワーク各層のドロップアウト率
    }
    SLOPE_THRESHOLD = 0.1  # ネットワーク修正の閾値
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    result_queue = Queue()
    weight_queue = Queue()

    # Workerプロセスの作成
    workers = []
    for worker_id in range(NUM_WORKERS):
        p = Process(target=follower_process, args=(ENV_NAME, result_queue, weight_queue, worker_id, NUM_ITERATIONS, DEVICE))
        workers.append(p)
        p.start()

    # Leaderプロセスの実行
    leader_process(result_queue, weight_queue, NUM_WORKERS, NUM_ITERATIONS, EMA_WINDOW, DROPOUT_RATES, SLOPE_THRESHOLD, DEVICE)

    # Workerプロセスの終了を待つ
    for p in workers:
        p.join()
