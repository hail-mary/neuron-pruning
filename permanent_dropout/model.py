import gymnasium as gym
import stable_baselines3
import torch
import os
import numpy as np
import cloudpickle

class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = gym.make(cfg['env_name'], render_mode=cfg['render_mode'])
        self.make_policy(self.env)
    
    @property
    def policy_kwargs(self):
        return self.model.policy_kwargs
    
    @property
    def policy(self):
        return self.model.policy
    
    def make_policy(self, env, policy_kwargs=None, policy_weights=None):
        if policy_kwargs is None: # to initialize
            policy_kwargs = self.cfg['policy_kwargs']

        if isinstance(policy_kwargs['activation_fn'], str): # load from cfg
            activation_fn = getattr(torch.nn, policy_kwargs['activation_fn'])
        else:
            activation_fn = policy_kwargs['activation_fn']

        policy_kwargs = dict(
            activation_fn=activation_fn,
            net_arch=policy_kwargs['net_arch']
        )
        algorithm = getattr(stable_baselines3, self.cfg['algorithm'])
        device = self.cfg['device']
        # self.model = None
        self.model = algorithm("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, device=device)
        if policy_weights is not None:
            self.model.policy.load_state_dict(policy_weights)

    def save_policy(self, save_to=''):
        policy_kwargs = self.model.policy_kwargs
        policy_weights = self.model.policy.state_dict()
        with open(f'{save_to}/policy_kwargs.pkl', 'wb') as f:
            cloudpickle.dump(policy_kwargs, f)
        with open(f'{save_to}/policy_weights.pkl', 'wb') as f:
            cloudpickle.dump(policy_weights, f)

    def load_policy(self, load_from=''):
        for file_name in os.listdir(load_from):
            if 'policy_kwargs.pkl' in file_name:
                with open(os.path.join(load_from, file_name), 'rb') as f:
                    deserialized_kwargs = cloudpickle.load(f)
            if 'policy_weights.pkl' in file_name:
                with open(os.path.join(load_from, file_name), 'rb') as f:
                    deserialized_weights = cloudpickle.load(f)

        self.make_policy(self.env, deserialized_kwargs, deserialized_weights)

    def learn(self, total_timesteps):
        self.model.learn(total_timesteps)

    def evaluate_policy(self, num_eval_episodes=5, num_eval_steps_per_episode=1000):
        total_rewards = []
        for _ in range(num_eval_episodes):
            observation, info = self.env.reset()
            episode_reward = 0
            for _ in range(num_eval_steps_per_episode):
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            total_rewards.append(episode_reward)
        self.env.close()
        return np.mean(total_rewards)
