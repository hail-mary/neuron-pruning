import os
import glob
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing import event_accumulator

def get_event_data(event_file, return_tags, flops_tags):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    available = ea.Tags()['scalars']
    
    ret_tag = next((t for t in return_tags if t in available), None)
    flp_tag = next((t for t in flops_tags if t in available), None)
    
    if not ret_tag or not flp_tag:
        return None, None, None, None

    return_events = ea.Scalars(ret_tag)
    flops_events = ea.Scalars(flp_tag)

    steps_return = np.array([e.step for e in return_events])
    values_return = np.array([e.value for e in return_events])
    
    steps_flops = np.array([e.step for e in flops_events])
    values_flops = np.array([e.value for e in flops_events])
    
    return steps_return, values_return, steps_flops, values_flops

def setup_plot_style():
    plt.rcParams['font.family'] = 'Times New Roman' if os.name == 'nt' else 'DejaVu Sans'
    plt.rcParams['font.size'] = 22
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['lines.linewidth'] = 3.0
    plt.rcParams["legend.framealpha"] = 1.0

def plot_learning_curve(log_dir, save_plot=False, save_legend=False):
    setup_plot_style()

    markers = ['s', '^', 'o']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    return_tags = ['eval/avg_return', 'eval/reward']
    flops_tags = ['eval/inference_flops', 'eval/flops', 'train/total_flops', 'total_flops']

    # Find methods
    methods = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    methods.sort()
    
    if not methods:
        print(f"No method directories found in {log_dir}")
        return

    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=120)
    ax2 = ax1.twinx()

    ax1.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    all_lns = []

    for i, method in enumerate(methods):
        method_path = os.path.join(log_dir, method)
        seed_dirs = glob.glob(os.path.join(method_path, "seed-*"))
        print(f"Method: {method}, Seeds found: {len(seed_dirs)}")
        
        if not seed_dirs:
            continue

        method_returns = []
        method_flops = []
        common_steps = None
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        for seed_dir in seed_dirs:
            event_files = glob.glob(os.path.join(seed_dir, "events.out.tfevents.*"))
            if not event_files:
                continue
            event_files.sort()
            event_file = event_files[-1]
            
            s_ret, v_ret, s_flp, v_flp = get_event_data(event_file, return_tags, flops_tags)
            
            if s_ret is None:
                continue
            
            if common_steps is None:
                common_steps = s_ret
            
            # Interpolate to common_steps to handle potential mismatches
            v_ret_interp = np.interp(common_steps, s_ret, v_ret)
            v_flp_interp = np.interp(common_steps, s_flp, v_flp)
            
            method_returns.append(v_ret_interp)
            method_flops.append(v_flp_interp)

        if not method_returns:
            continue
            
        mean_returns = np.mean(method_returns, axis=0)
        mean_flops = np.mean(method_flops, axis=0)
        std_returns = np.std(method_returns, axis=0)
        
        # Plotting Return on Left axis (Solid line)
        ln1, = ax1.plot(common_steps, mean_returns, color=color, marker=marker, 
                        label=f'{method} Return', linewidth=2.5, linestyle='-', 
                        markersize=8, markevery=max(1, len(common_steps)//15))
        # Optional: shaded area for std
        ax1.fill_between(common_steps, mean_returns - std_returns, mean_returns + std_returns, color=color, alpha=0.2)
        
        # Plotting FLOPs on Right axis (Dashed line)
        ln2, = ax2.plot(common_steps, mean_flops, color=color, marker=marker, 
                        label=f'{method} FLOPs', linewidth=2.5, linestyle='--',
                        markersize=8, markevery=max(1, len(common_steps)//15))
        
        all_lns.extend([ln1, ln2])

        # Calculate mean and std of the last 10 evaluation values across all seeds
        last_10_all = np.concatenate([v[-10:] for v in method_returns])
        final_mean = np.mean(last_10_all)
        final_std = np.std(last_10_all)
        print(f"{method}: {final_mean:.2f} +/- {final_std:.2f}")

    ax1.set_xlabel('Iteration', fontweight='bold')
    ax1.set_ylabel('Episode Return', fontweight='bold')
    ax2.set_ylabel('Inference FLOPs', fontweight='bold')
    
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Combined Legend
    labs = [l.get_label() for l in all_lns]
    # ax1.legend(all_lns, labs, loc='upper left', frameon=True, framealpha=0.9, edgecolor='#ddd', fontsize=14)

    plt.tight_layout()
    
    if save_legend:
        # Create a separate figure for the legend
        fig_leg = plt.figure(figsize=(10, 2))
        ax_leg = fig_leg.add_subplot(111)
        ax_leg.axis('off')
        
        # Calculate number of columns for legend
        ncol = len(methods)
        legend = ax_leg.legend(all_lns, labs, loc='center', ncol=ncol, 
                               frameon=True, framealpha=1.0, edgecolor='#ddd', fontsize=18)
        
        # Draw canvas to get dimensions
        fig_leg.canvas.draw()
        
        # Get legend bounding box and expand slightly for padding
        bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
        # Add small padding (roughly 0.1 inches)
        padding = 0.1
        bbox_padded = bbox.expanded(1.1, 1.1) 
        
        legend_path = f"legend_{os.path.basename(log_dir)}.pdf"
        fig_leg.savefig(legend_path, dpi=300, bbox_inches=bbox_padded)
        print(f"Legend saved to {legend_path}")
        plt.close(fig_leg)

    if save_plot:
        output_path = f"learning_curve_{os.path.basename(log_dir)}.pdf"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()

def plot_from_json(json_files, plot1_every=100, plot2_every=10, labels=None):
    setup_plot_style()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    if labels is None:
        labels = [os.path.basename(f).replace('.json', '') for f in json_files]
        
    markers = ['s', '^', 'o']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markersize = 9

    for i, json_file in enumerate(json_files):
        with open(json_file, 'r') as f:
            history = json.load(f)

        iterations = history['iterations'][::plot1_every] + [history['iterations'][-1]]
        mean_rewards = history['mean_rewards'][::plot1_every] + [history['mean_rewards'][-1]]
        std_rewards = history['std_rewards'][::plot1_every] + [history['std_rewards'][-1]]
        
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]

        ax1.plot(iterations, mean_rewards, label=labels[i], marker=m, markersize=markersize, color=c)
        ax1.fill_between(iterations, 
                         np.array(mean_rewards) - np.array(std_rewards), 
                         np.array(mean_rewards) + np.array(std_rewards), 
                         alpha=0.2, color=c)
           
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Episode Return')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()
    for i, json_file in enumerate(json_files):
        with open(json_file, 'r') as f:
            history = json.load(f)
        
        iterations2 = history['iterations'][::plot2_every]
        flops_counts = (history.get('inference_flops') or history.get('flops_counts'))[::plot2_every]
        
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        ax2.plot(iterations2, flops_counts, linestyle='--', marker=m, markevery=10, markersize=markersize, color=c)

    ax2.set_ylabel('Inference FLOPs')
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    ax1.legend(loc='lower left')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=r"Ant-v5", help="Environment name")
    parser.add_argument("--logdir", type=str, default=r"data", help="Directory containing method/seed-* subdirectories")
    parser.add_argument("--save", action="store_true", help="Save the plot as PDF")
    parser.add_argument("--save_legend", action="store_true", help="Save the legend separately as PDF")
    parser.add_argument("--json", type=str, nargs='+', help="Plot from JSON history files")
    args = parser.parse_args()

    if args.json:
        plot_from_json(args.json)
    else:
        logdir = os.path.join(args.logdir, args.env)
        plot_learning_curve(logdir, save_plot=args.save, save_legend=args.save_legend)

