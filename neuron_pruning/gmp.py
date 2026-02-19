import gymnasium as gym
import numpy as np
import torch
import torch.nn.utils.prune as prune
from stable_baselines3 import PPO

def eval_policy(model, num_episodes=5):
    rewards = []
    env = gym.make("Ant-v5", render_mode="rgb_array")
    for _ in range(num_episodes):
        total_reward = 0
        obs, _ = env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                obs, _ = env.reset()
        rewards.append(total_reward)
    env.close()
    return np.mean(rewards)

def calculate_sparsity(tensor):
    """Calculates the sparsity of a tensor."""
    return 100. * float(torch.sum(tensor == 0)) / float(tensor.numel())

def apply_gmp(model, amount=0.2):
    """
    Apply Global Magnitude Pruning to the model.
    
    Args:
        model (PPO): The PPO model to prune.
        amount (float): The fraction of parameters to prune.
    """
    for name, module in model.policy.named_modules():
        if ("policy" in name or "action" in name) and isinstance(module, torch.nn.Linear):
            parameters_to_prune = [(module, name) for name, parameter in module.named_parameters()]
            prune.global_unstructured(
                parameters=parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )
            # Optionally remove the pruning reparameterization
            prune.remove(module, 'weight')
            if hasattr(module, 'bias'):
                prune.remove(module, 'bias')  # Remove bias pruning if it exists

def gradual_pruning(model, current_step, total_steps, final_sparsity=0.8):
    """
    Gradually prune the model based on the current step.
    
    Args:
        model (PPO): The PPO model to prune.
        current_step (int): The current training step.
        total_steps (int): The total number of training steps.
        final_sparsity (float): The final sparsity level to achieve.
    """
    target_sparsity = final_sparsity * (1 - (1 - ((current_step)/ total_steps)) ** 3)
    apply_gmp(model, amount=target_sparsity)
    print("Current sparsity levels after pruning:")
    for name, module in model.policy.named_modules():
        if ("policy" in name or "action" in name) and isinstance(module, torch.nn.Linear):
             print(f"{name}.weight: {calculate_sparsity(module.weight):.2f}%")



env_id = "Ant-v5"
env = gym.make(env_id, render_mode="rgb_array")
policy_kwargs = dict(
    net_arch=[256, 256],
    activation_fn=torch.nn.ReLU,
)
final_sparsity = 0.7
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0, tensorboard_log=f"tensorboard_logs/{env_id}_gmp{final_sparsity:.2f}")
# print(model.policy_kwargs)
# total_steps = 1000
# prune_start = int(0.2 * total_steps)
# prune_end = int(0.8 * total_steps)

# rewards = []
# for step in range(total_steps):
#     model.learn(total_timesteps=1000, reset_num_timesteps=False)
#     rewards.append(eval_policy(model, num_episodes=5))
#     print(f"\nStep {step + 1}/{total_steps}, Reward: {rewards[-1]:.2f}")
#     if step in np.arange(prune_start, prune_end):
#         gradual_pruning(model, step, total_steps, final_sparsity=final_sparsity)

# model.save(f"models/ant_{final_sparsity:.2f}pruned")

model = PPO.load(f"models/ant_0.70pruned", env=env, policy_kwargs=policy_kwargs)
# for key, val in model.policy.state_dict().items():
#   if "policy" in key or "action" in key:
#     print(key, val)
for name, module in model.policy.named_modules():
    if ("policy" in name or "action" in name) and isinstance(module, torch.nn.Linear):
            print(f"{name}.weight: {module.weight}")
