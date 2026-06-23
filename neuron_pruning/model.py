import gymnasium as gym
import stable_baselines3
import torch
import os
import numpy as np
import cloudpickle

class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg['render_mode'] == 'None':
            cfg['render_mode'] = None
        self.env = gym.make(cfg['env_name'], render_mode=cfg['render_mode'])
        self.env.reset(seed=cfg['seed'])
        self.make_policy(self.env)
    
    @property
    def policy_kwargs(self):
        return self.model.policy_kwargs
    
    @property
    def policy(self):
        return self.model.policy
    
    def make_policy(self, env, policy_kwargs=None, policy_weights=None, seed=None, optimizer_state=None):
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
        if seed is None:
            seed = self.cfg['seed']
        self.model = algorithm("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, device=device, n_steps=self.cfg['timesteps_per_iteration'], seed=seed)
        
        if policy_weights is not None:
            self.model.policy.load_state_dict(policy_weights)
        
        if optimizer_state is not None:
            self.load_optimizer_state(optimizer_state)

    def load_optimizer_state(self, state_to_load):
        """
        Custom loader to map state_dict back to the optimizer.
        state_to_load: dict mapping parameter names (e.g. 'mlp_extractor...') to state dicts (e.g. {'exp_avg': ...})
        """
        params_dict = dict(self.model.policy.named_parameters())
        optimizer = self.model.policy.optimizer
        
        # Build a mapping from parameter object to state
        for name, param in params_dict.items():
            if name in state_to_load:
                optimizer.state[param] = state_to_load[name]

    def get_optimizer_state_by_name(self):
        """
        Returns a dictionary mapping parameter names to their optimizer states.
        """
        params_dict = dict(self.model.policy.named_parameters())
        optimizer = self.model.policy.optimizer
        state_by_name = {}
        
        for name, param in params_dict.items():
            if param in optimizer.state:
                state_by_name[name] = optimizer.state[param]
        
        return state_by_name

    def save_policy(self, save_to=''):
        policy_kwargs = self.model.policy_kwargs
        policy_weights = self.model.policy.state_dict()
        with open(f'{save_to}/policy_kwargs.pkl', 'wb') as f:
            cloudpickle.dump(policy_kwargs, f)
        with open(f'{save_to}/policy_weights.pkl', 'wb') as f:
            cloudpickle.dump(policy_weights, f)

    def load_policy(self, load_from=''):
        deserialized_kwargs = None
        deserialized_weights = None
        
        # First, try to find files in the specified directory
        if os.path.exists(load_from):
            for file_name in os.listdir(load_from):
                if 'policy_kwargs.pkl' in file_name:
                    with open(os.path.join(load_from, file_name), 'rb') as f:
                        deserialized_kwargs = cloudpickle.load(f)
                if 'policy_weights.pkl' in file_name:
                    with open(os.path.join(load_from, file_name), 'rb') as f:
                        deserialized_weights = cloudpickle.load(f)

        # If not found, look for Iteration-X subdirectories and take the latest one
        if (deserialized_kwargs is None or deserialized_weights is None) and os.path.exists(load_from):
            subdirs = [d for d in os.listdir(load_from) if os.path.isdir(os.path.join(load_from, d)) and 'Iteration-' in d]
            if subdirs:
                # Sort by iteration index (assumes format Iteration-N)
                subdirs.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)
                latest_subdir = os.path.join(load_from, subdirs[0])
                print(f"Loading from latest checkpoint: {latest_subdir}")
                for file_name in os.listdir(latest_subdir):
                    if 'policy_kwargs.pkl' in file_name:
                        with open(os.path.join(latest_subdir, file_name), 'rb') as f:
                            deserialized_kwargs = cloudpickle.load(f)
                    if 'policy_weights.pkl' in file_name:
                        with open(os.path.join(latest_subdir, file_name), 'rb') as f:
                            deserialized_weights = cloudpickle.load(f)

        if deserialized_kwargs is None or deserialized_weights is None:
            raise FileNotFoundError(f"Could not find policy files in {load_from} or its subdirectories.")

        self.make_policy(self.env, deserialized_kwargs, deserialized_weights)

    def learn(self, total_timesteps):
        self.model.learn(total_timesteps)

    def evaluate_policy(self, seed=None, num_eval_episodes=1, num_eval_steps_per_episode=1000, record_video=False):
        total_rewards = []
        
        # Wrap the environment for video recording if record_video is True
        if record_video:
            video_folder = os.path.join(self.cfg['logdir'], 'videos')
            self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder, episode_trigger=lambda x: True)
        
        for i in range(num_eval_episodes):
            observation, info = self.env.reset(seed=seed+i)
            episode_reward = 0
            for _ in range(num_eval_steps_per_episode):
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break 
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards), np.std(total_rewards)
