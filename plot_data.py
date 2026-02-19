import os
import glob
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_learning_curve(log_dir):
    # Find event files in the directory
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        # If not found, check if log_dir itself is an event file
        if os.path.isfile(log_dir) and "events.out.tfevents" in log_dir:
            event_file = log_dir
        else:
            print(f"No event files found in {log_dir}")
            return
    else:
        # Take the first one (or the latest by sorting)
        event_files.sort()
        event_file = event_files[-1]

    print(f"Reading {event_file}...")
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    # Define tags
    return_tag = 'eval/avg_return'
    flops_tag = 'eval/total_flops'

    try:
        return_events = ea.Scalars(return_tag)
        flops_events = ea.Scalars(flops_tag)
    except KeyError as e:
        print(f"Tag not found: {e}")
        print("Available tags:", ea.Tags()['scalars'])
        return

    # Extract data
    steps_return = [e.step for e in return_events]
    values_return = [e.value for e in return_events]
    
    steps_flops = [e.step for e in flops_events]
    values_flops = [e.value for e in flops_events]

    # Plotting
    plt.rcParams['font.family'] = 'Times New Roman' if os.name == 'nt' else 'DejaVu Sans'
    plt.rcParams['font.size'] = 22
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['grid.alpha'] = 0.3

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=120)

    # Gradient-like background feel or just clean white
    ax1.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    # Plot Episode Return on Left Y-axis
    color_return = '#1a73e8' # Google Blue
    ax1.set_xlabel('Iteration', fontweight='bold')
    ax1.set_ylabel('Episode Return', color=color_return, fontweight='bold')
    lns1 = ax1.plot(steps_return, values_return, color=color_return, label='Episode Return', linewidth=2.5, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color_return)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Total FLOPs on Right Y-axis
    ax2 = ax1.twinx()
    color_flops = '#d93025' # Google Red
    ax2.set_ylabel('Total FLOPs', color=color_flops, fontweight='bold')
    lns2 = ax2.plot(steps_flops, values_flops, color=color_flops, label='Total FLOPs', linewidth=2.5, linestyle='--', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color_flops)

    # Combine legends
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', frameon=True, framealpha=0.9, edgecolor='#ddd')

    plt.title(f'Learning Curve: {os.path.basename(log_dir)}', pad=20, fontweight='bold')
    
    # Save the plot
    output_path = "learning_curve.pdf"
    plt.tight_layout()
    plt.show()
    # plt.savefig(output_path, bbox_inches='tight')
    # print(f"Plot saved to {output_path}")
    
    # Show plot if possible
    # plt.show() # Disabled for headless/script run, but user might want it

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=r"logs_avg4\seed0", help="Directory containing event files")
    args = parser.parse_args()

    # The user specifically mentioned logs_avg4\seed0\events.out.tfevents.1771384472.DESKTOP-AUQBIB9.18268.0
    # I'll check if the input is the directory or the file.
    plot_learning_curve(args.log_dir)