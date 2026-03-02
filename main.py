import yaml
import argparse
import numpy as np
import torch
import random
from multiprocessing import Process, Queue
import time

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
            if model.policy_kwargs['net_arch'] != new_arch:  # Net architecture has been changed
                new_kwargs = model.policy_kwargs.copy()
                new_kwargs['net_arch'] = new_arch
                model.make_policy(env, new_kwargs, new_params, worker_seed)
            else:
                model.make_policy(env, model.policy_kwargs, new_params, worker_seed)

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

    best_policy_arch = None
    best_reward = float('-inf')
    best_iteration = 0

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
        total_flops = flops_pi + flops_vf
        
        # Log statistics
        new_best_reward = max(all_rewards)

        # Track the best policy architecture
        if new_best_reward > best_reward:
            best_reward = new_best_reward
            best_policy_arch = policy_arch.copy()
            best_iteration = iteration
            should_save = True

        # results = [(0, mean_reward, policy_kwargs, avg_params)]
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
        writer.add_scalar('eval/total_flops', total_flops, iteration)

        # Output: architecture information
        print("Policy Summary:")
        for layer, shape in policy_arch.items():
            if layer == 'pi':
                flops = flops_pi
            else:
                flops = flops_vf
            print(f"  - {layer}: {shape} FLOPs: {flops}")

        if iteration > 0 and iteration % cfg['update_interval'] == 0:
            print(f"\n----------- Iteration {iteration}/{num_iterations}: Modifying network architecture ----------")
            arch, params, aux = scheduler.preprocess(raw_arch=policy_arch, raw_params=avg_params)
            modified_arch, modified_params = scheduler.modify_network(params, arch, iteration, target_sparsity)
            modified_arch, modified_params = scheduler.reconstruct(modified_arch, modified_params, aux)

        else:
            modified_arch = policy_arch
            modified_params = avg_params

        # Record neuron count
        # total_neurons = sum(sum(units) for units in modified_arch.values())
        # neuron_counts.append(total_neurons)
        total_flops = flops_pi + flops_vf
        flops_counts.append(total_flops)

        # Send weights and new architecture to each worker
        for worker_id in range(num_workers):
            network_queue.put((modified_arch, modified_params))

    # End timing the training
    end_time = time.time()

    # Calculate average reward
    average_reward = sum(all_rewards) / len(all_rewards)

    # Log the training summary using the Logger class
    logger.log_training_summary(start_time, end_time, best_reward, average_reward, best_policy_arch, best_iteration)

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
    parser.add_argument('--plot', type=str, nargs='+', help='Plot learning curve from the specified JSON file(s).')
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

    if args.eval:
        import pathlib
        cfg['logdir'] = pathlib.Path(args.eval).parent.parent
        # Evaluate the policy
        eval_only(cfg, load_from=args.eval, record_video=args.record)

    elif args.plot:
        # Plot learning curve from the specified JSON file
        logger = Logger(cfg, save_cfg=False)
        logger.plot_learning_curve_from_json(args.plot)

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

        # Leader process
        leader_process(result_queue, network_queue, cfg)

        # Wait for worker processes to finish
        for p in workers:
            p.join()

if __name__ == "__main__":
    main()
