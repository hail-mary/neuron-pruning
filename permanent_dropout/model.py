import gymnasium as gym
import stable_baselines3
import torch
import numpy as np
import cloudpickle

class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        env = gym.make(cfg['env_name'], render_mode=cfg['render_mode'])
        policy_kwargs = cfg['policy_kwargs']
        activation_fn = getattr(torch.nn, policy_kwargs['activation_fn'])
        policy_kwargs = dict(
            activation_fn=activation_fn,
            net_arch=policy_kwargs['net_arch']
        )
        algorithm = getattr(stable_baselines3, cfg['algorithm'])
        device = cfg['device']

        self.model = algorithm("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, device=device)
        return self.model
    
    def save_policy(self):
        pass

    def load_policy(self):
        pass

    def learn(self, total_timesteps):
        pass

    def evaluate_policy(self, num_eval_episodes=5, num_eval_steps_per_episode=1000):
        env = self.cfg['env_name']
        total_rewards = []
        for _ in range(num_eval_episodes):
            observation, info = env.reset()
            episode_reward = 0
            for _ in range(num_eval_steps_per_episode):
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)
