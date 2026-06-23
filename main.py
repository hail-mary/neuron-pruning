import yaml
import argparse
import numpy as np
import torch
import random
from multiprocessing import Process, Queue
import time
import json

from neuron_pruning.model import Model
from neuron_pruning.scheduler import Scheduler
from neuron_pruning.logger import Logger
from torch.utils.tensorboard import SummaryWriter

def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def eval_only(cfg, load_from, record_video=False):
    # Load the model
    model = Model(cfg)
    model.load_policy(load_from)  # Assuming you have a method to load the trained policy

    # Evaluate the policy
    print('\n#------------------- Start Evaluation ! ---------------------#')
    total_reward, _ = model.evaluate_policy(record_video=record_video)
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

        # Averaging and architecture sync
        if iteration % cfg.get('average_interval', 1) == 0 or iteration == cfg['num_iterations'] - 1:
            # Send current policy params and architecture to the leader
            result_queue.put((worker_id, model.policy_kwargs, model.policy.state_dict()))

            # Receive new policy params and architecture from the leader
            new_arch, new_params = network_queue.get()

            # Each iteration/worker combination gets a unique seed offset from the base seed
            worker_seed = cfg['seed'] + iteration * cfg['num_workers'] + worker_id
            # Update architecture and unconditionally reset optimizer state upon WA
            new_kwargs = model.policy_kwargs.copy()
            new_kwargs['net_arch'] = new_arch
            model.make_policy(env, new_kwargs, new_params, worker_seed)

def leader_process(result_queue, network_queue, cfg):
    """
    The leader collects evaluation results and modify architecture.
    """
    num_workers = cfg['num_workers']
    num_iterations = cfg['num_iterations']
    target_sparsity = cfg['target_sparsity']
    scheduler = Scheduler(cfg)
    logger = Logger(cfg)
    writer = SummaryWriter(cfg['logdir'])
    global_model = Model(cfg)

    all_rewards = []
    rewards_per_worker = [[] for _ in range(num_workers)]  # Record rewards for each worker
    # neuron_counts = []  # Track neuron count changes
    flops_counts = []
    cumulative_training_flops = 0

    best_policy_arch = None
    best_reward = float('-inf')
    best_iteration = 0

    # Calculate initial FLOPs for percentage display
    initial_params_pi, initial_params_vf = calc_params(cfg['policy_kwargs']['net_arch'], cfg)
    initial_flops_pi = initial_params_pi * 2
    initial_flops_vf = initial_params_vf * 2

    print('\n#---------------------- Start Training ! -----------------------#')
    # Start timing the training
    start_time = time.time()
    for iteration in range(num_iterations):
        terminate = iteration == num_iterations - 1
        
        # Check if we should perform averaging and evaluation
        if iteration % cfg.get('average_interval', 1) != 0 and not terminate:
            continue
            
        should_save = False
        
        params = []
        policy_arch = None
        logger.step() 

        # Collect results from each worker
        for _ in range(num_workers):
            worker_id, policy_kwargs, policy_weight = result_queue.get()
            params.append(policy_weight)
            policy_arch = policy_kwargs["net_arch"]

        # Average weights
        avg_params = scheduler.average_params(params)
        
        # Load into global model for evaluation
        if global_model.policy_kwargs['net_arch'] != policy_arch:
            global_model.make_policy(global_model.env, policy_kwargs, avg_params)
        else:
            global_model.model.policy.load_state_dict(avg_params)
        
        # Evaluate global model
        # Use a high offset from global seed for evaluation to avoid overlapping with training seeds
        mean_reward, std_reward = global_model.evaluate_policy(seed=cfg['seed'] + 1000000, num_eval_episodes=10)
        all_rewards.append(mean_reward)
        for i in range(num_workers):
            rewards_per_worker[i].append(mean_reward)

        params_pi, params_vf = calc_params(policy_arch, cfg)
        flops_pi, flops_vf = params_pi * 2, params_vf * 2
        inference_flops = flops_pi + flops_vf
        
        # Training FLOPs for this iteration
        n_epochs = 10 
        rollout_cost = num_workers * cfg['timesteps_per_iteration'] * inference_flops
        update_cost = num_workers * cfg['timesteps_per_iteration'] * (n_epochs * 3 * inference_flops)
        averaging_cost = num_workers * (params_pi + params_vf)
        
        iteration_training_flops = rollout_cost + update_cost + averaging_cost
        cumulative_training_flops += iteration_training_flops

        # Log statistics
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_policy_arch = policy_arch.copy()
            best_iteration = iteration
            should_save = True

        results = [(0, mean_reward, policy_kwargs, avg_params)]
        # if should_save or terminate:
        #     logger.save_checkpoint(results)
        #     should_save = False

        # Output: statistics
        print(f"\n==== Iteration {iteration+1}/{num_iterations} ==== ")
        elapsed = time.time() - start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        print(f"Return Avg: {mean_reward:.2f}, Std: {std_reward:.2f}, Elapsed: {h:02d}:{m:02d}:{s:02d}")
        writer.add_scalar('eval/avg_return', mean_reward, iteration)
        writer.add_scalar('eval/std_return', std_reward, iteration)
        writer.add_scalar('eval/inference_flops', inference_flops, iteration)
        writer.add_scalar('eval/iteration_training_flops', iteration_training_flops, iteration)
        writer.add_scalar('eval/cumulative_training_flops', cumulative_training_flops, iteration)

        # Output: architecture information
        print(f"Inference FLOPs (per step): {inference_flops}")
        print(f"Training FLOPs (cumulative): {cumulative_training_flops:.2e}")
        print("Policy Summary:")
        for layer, shape in policy_arch.items():
            if layer == 'pi':
                flops = flops_pi
                initial_flops = initial_flops_pi
            else:
                flops = flops_vf
                initial_flops = initial_flops_vf
            ratio = (flops / initial_flops) * 100 if initial_flops > 0 else 0
            print(f"  - {layer}: {shape} FLOPs: {flops} ({ratio:.2f}%)")

        if iteration > 0 and iteration % cfg['update_interval'] == 0:
            print(f"\n----------- Iteration {iteration}/{num_iterations}: Modifying network architecture ----------")
            arch, params, aux = scheduler.preprocess(raw_arch=policy_arch, raw_params=avg_params)
            modified_arch, modified_params, _ = scheduler.modify_network(params, arch, iteration, target_sparsity)
            modified_arch, modified_params = scheduler.reconstruct(modified_arch, modified_params, aux)
        else:
            modified_arch = policy_arch
            modified_params = avg_params

        # Log the data for each iteration
        logger.log_iteration(mean_reward, std_reward, inference_flops, iteration_training_flops)

        # Send weights and new architecture to each worker
        for worker_id in range(num_workers):
            network_queue.put((modified_arch, modified_params))

    # End timing the training
    end_time = time.time()
    
    # Save the last model explicitly for post-training profiling
    last_results = [(0, mean_reward, policy_kwargs, avg_params)]
    logger.save_checkpoint(last_results, save_to='last_model')

    # Save the history to a JSON file for analysis
    logger.save_history()

    # Log the training summary using the Logger class
    logger.log_training_summary(
        start_time, end_time, all_rewards, 
        best_policy_arch, best_iteration, 
        cumulative_training_flops=cumulative_training_flops
    )

def calc_params(arch, cfg):
    obs_dim = cfg['obs_dim']
    action_dim = cfg['action_dim']
    arch_pi = [obs_dim] + arch['pi'] + [action_dim]
    arch_vf = [obs_dim] + arch['vf'] + [1]
    params_pi = 0
    params_vf = 0
    for i in range(len(arch_pi) - 1):
        params_pi += arch_pi[i] * arch_pi[i+1]
    for i in range(len(arch_vf) - 1):
        params_vf += arch_vf[i] * arch_vf[i+1]
    
    return params_pi, params_vf

def profile_inference(cfg, load_from=None):
    """
    Independent profiling of the current or a loaded architecture for inference.
    Calculates FLOPs and measures latency for actor and critic separately.
    """
    from neuron_pruning.model import Model
    import time
    import os

    # Setup model
    model_wrapper = Model(cfg)
    
    if load_from:
        print(f"Loading model from {load_from} for profiling...")
        model_wrapper.load_policy(load_from)
    
    policy = model_wrapper.policy
    obs_dim = cfg['obs_dim']
    dummy_input = torch.randn(1, obs_dim).to(cfg['device'])
    
    # Calculate FLOPs based on the ACTUAL architecture being profiled
    current_arch = model_wrapper.policy_kwargs['net_arch']
    params_pi, params_vf = calc_params(current_arch, cfg)
    flops_pi = params_pi * 2
    flops_vf = params_vf * 2
    
    # Latency measurement helper
    def measure_latency(func, input_data, n_warmup=100, n_trials=1000):
        # Warmup
        for _ in range(n_warmup):
            _ = func(input_data)
        
        # Timing
        if cfg['device'] != 'cpu' and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        for _ in range(n_trials):
            _ = func(input_data)
        
        if cfg['device'] != 'cpu' and torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        return (end_time - start_time) / n_trials * 1000  # ms per inference

    # Measure latency for Actor (pi)
    # Actor in SB3 involves mlp_extractor.policy_net and action_net
    # We use get_distribution as it's the standard path for forward pass in PPO
    latency_pi = measure_latency(lambda x: policy.get_distribution(x), dummy_input)
    
    # Measure latency for Critic (vf)
    # Critic in SB3 involves mlp_extractor.value_net and value_net
    latency_vf = measure_latency(lambda x: policy.predict_values(x), dummy_input)

    print("\n#----------------- Inference Profiling -------------------#")
    print(f"Architecture: {current_arch}")
    print(f"Device: {cfg['device']}")
    print("-" * 50)
    print(f"Actor (pi):")
    print(f"  FLOPs:   {flops_pi}")
    print(f"  Latency: {latency_pi:.4f} ms")
    print("-" * 50)
    print(f"Critic (vf):")
    print(f"  FLOPs:   {flops_vf}")
    print(f"  Latency: {latency_vf:.4f} ms")
    print("-" * 50)

    print(f"Total Inference FLOPs: {flops_pi + flops_vf}")
    print(f"Total Inference Latency: {latency_pi + latency_vf:.4f} ms")
    print("#---------------------------------------------------------#\n")

    return {
        "architecture": current_arch,
        "actor": {"flops": flops_pi, "latency_ms": latency_pi},
        "critic": {"flops": flops_vf, "latency_ms": latency_vf}
    }

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train or evaluate the policy.')
    parser.add_argument('--cfg', type=str, default='config.yaml', help='Path to the configuration file.')
    parser.add_argument('--env', type=str, help='Specify the environment name to override the config file.')
    parser.add_argument('--eval', type=str, help='Only evaluate a trained policy. Specify the directory to load the policy from.')
    parser.add_argument('--logdir', type=str, help='Specify the directory for logging.')
    parser.add_argument('--seed', type=int, help='Specify the seed to override the config file.')
    parser.add_argument('--update_interval', type=int, help='Specify the update interval to override the config file.')
    parser.add_argument('--average_interval', type=int, help='Specify the average interval to override the config file.')
    parser.add_argument('--num_iterations', type=int, help='Specify the number of iterations to override the config file.')
    parser.add_argument('--record', action='store_true', help='Record video of the best model during evaluation.')
    parser.add_argument('--profile', action='store_true', help='Profile the inference cost of the architecture specified in the config.')
    args = parser.parse_args()

     # Load configuration
    if args.cfg:
        cfg = load_config(args.cfg) 
    else:
        cfg = load_config('config.yaml') # Use the --cfg argument to load the configuration file

    # Override env_name if specified
    if args.env:
        cfg['env_name'] = args.env
    
    import os
    if args.logdir:
        cfg['logdir'] = args.logdir
    
    if args.seed is not None:
        cfg['seed'] = args.seed
    
    if args.update_interval:
        cfg['update_interval'] = args.update_interval
    
    if args.average_interval:
        cfg['average_interval'] = args.average_interval
    
    if args.num_iterations:
        cfg['num_iterations'] = args.num_iterations
    
    # Append seed to logdir for unique run directories
    cfg['logdir'] = os.path.join(cfg['logdir'], f"seed-{cfg['seed']}")

    # Set global seeds for reproducibility
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if cfg['device'] != 'cpu' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Assert if dropout rates and net_arch are consistent
    assert len(cfg['target_sparsity']['policy']) == len(cfg['policy_kwargs']['net_arch']['pi'])
    if cfg['algorithm'] == 'PPO':
        assert len(cfg['target_sparsity']['value']) == len(cfg['policy_kwargs']['net_arch']['vf'])
    else:
        assert len(cfg['target_sparsity']['value']) == len(cfg['policy_kwargs']['net_arch']['qf'])

    if args.profile:
        # Profile inference only (optionally loading a specific model via --eval)
        import gymnasium as gym
        env = gym.make(cfg['env_name'], render_mode=None)
        cfg['obs_dim'] = env.observation_space.shape[0]
        cfg['action_dim'] = env.action_space.shape[0]
        profile_inference(cfg, load_from=args.eval)

    elif args.eval:
        import pathlib
        cfg['logdir'] = pathlib.Path(args.eval).parent.parent
        # Evaluate the policy
        eval_only(cfg, load_from=args.eval, record_video=args.record)


    else:
        import gymnasium as gym
        env = gym.make(cfg['env_name'], render_mode=None)
        cfg['obs_dim'] = env.observation_space.shape[0]
        cfg['action_dim'] = env.action_space.shape[0]
        # Training
        result_queue = Queue()
        network_queue = Queue()

        # Worker processes
        workers = []
        for worker_id in range(cfg['num_workers']):
            p = Process(target=follower_process, args=(result_queue, network_queue, worker_id, cfg))
            workers.append(p)
            p.start()

        # Pre-training profiling
        print("\n[Pre-training Profiling]")
        pre_profile = profile_inference(cfg)

        # Leader process
        leader_process(result_queue, network_queue, cfg)

        # Post-training profiling (with last model)
        print("\n[Post-training Profiling]")
        last_model_dir = os.path.join(cfg['logdir'], "last_model")
        # Find the latest iteration subdirectory
        post_profile = None
        if os.path.exists(last_model_dir):
            subdirs = [d for d in os.listdir(last_model_dir) if os.path.isdir(os.path.join(last_model_dir, d)) and 'Iteration-' in d]
            if subdirs:
                subdirs.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)
                latest_subdir = os.path.join(last_model_dir, subdirs[0])
                print(f"Profiling last model from {latest_subdir}")
                post_profile = profile_inference(cfg, load_from=latest_subdir)
        
        if post_profile is None:
            # Fallback to the current memory state if file not found
            post_profile = profile_inference(cfg)

        # Save profiling results to JSON
        profiling_results = {
            "pre_training": pre_profile,
            "post_training": post_profile
        }
        with open(os.path.join(cfg['logdir'], 'profiling_results.json'), 'w') as f:
            json.dump(profiling_results, f, indent=4)

        # Wait for worker processes to finish
        for p in workers:
            p.join()

if __name__ == "__main__":
    main()
