#%%
import glob
import statistics

# Specify the string that should be contained in the directory name
method = "ord"
env = "ant"

# Use glob to find all matching files
file_pattern = f"./ordinary_RL_less_nodes_89_56/{method}*{env}*/training_summary.txt"
files = glob.glob(file_pattern)

# Lists to store the extracted values
total_training_times = []
best_rewards = []

# Iterate over each file and extract the required information
for file_path in files:
    with open(file_path, 'r') as file:
        total_training_time = None
        best_reward = None
        for line in file:
            if "Total Training Time" in line:
                total_training_time = float(line.split(":")[-1].strip().replace(" hours", ""))
            elif "Best Reward" in line:
                best_reward = float(line.split(":")[-1].strip())
        
        # Store the extracted information
        if total_training_time is not None and best_reward is not None:
            total_training_times.append(total_training_time)
            best_rewards.append(best_reward)

# Calculate and print mean and standard deviation
if total_training_times and best_rewards:
    mean_training_time = statistics.mean(total_training_times)
    std_training_time = statistics.stdev(total_training_times)
    mean_best_reward = statistics.mean(best_rewards)
    std_best_reward = statistics.stdev(best_rewards)

    print(f"Method: {method}")
    print(f"Env: {env}")
    print(f"Data: {len(total_training_times)}")
    print(f"Mean Total Training Time: {mean_training_time:.2f}")
    print(f"Standard Deviation of Total Training Time: {std_training_time:.2f}")
    print(f"Mean Best Reward: {mean_best_reward:.2f}")
    print(f"Standard Deviation of Best Reward: {std_best_reward:.2f}")

#%%
from scipy import stats
import numpy as np

def non_inferiority_test(conventional, proposed, margin=-100, alpha=0.05):
    """
    非劣性検定（one-sided t-test）
    H0: μ_proposed - μ_conventional ≤ margin（非劣性なし）
    H1: μ_proposed - μ_conventional > margin（非劣性あり）

    Parameters:
        conventional: list or array of conventional method results
        proposed: list or array of proposed method results
        margin: non-inferiority margin (default: -100)
        alpha: significance level (default: 0.05)

    Returns:
        Dictionary with t_stat, p_value, mean_diff, standard_error, and conclusion
    """

    if len(conventional) != len(proposed):
        raise ValueError("データ数が一致していません")

    # 差分
    differences = np.array(proposed) - np.array(conventional)
    mean_diff = np.mean(differences)
    std_err = stats.sem(differences)
    df = len(differences) - 1

    # t統計量とp値（片側）
    t_stat = (mean_diff - margin) / (std_err + 1e-8)
    p_value = 1 - stats.t.cdf(t_stat, df)

    # 結論
    conclusion = "非劣性が統計的に示された" if p_value < alpha else "非劣性は統計的に有意でない"

    return {
        "mean_difference": mean_diff,
        "standard_error": std_err,
        "t_statistic": t_stat,
        "p_value": p_value,
        "conclusion": conclusion
    }
# ant 
margin = -150
less_nodes = [3734.2516004691797, 4552.022940332504, 6207.9310014736575, 4987.688325494861, 3773.625000583833, 4176.278348469197, 6080.06725660151, 6265.115960681642, 6259.193447606352, 6025.181562523217]
conventional = [4773.332105428182, 4195.215504842041, 6202.7134097303615, 5700.535772062887, 6145.921148769905, 6157.979834067982, 6036.689140882168, 6063.157940676491, 6033.074070155769, 4160.7889726086405]
proposed =     [4162.414867656757, 5977.7392207798885, 6280.978099271086, 5942.774371653629, 5996.557126260057, 5905.7751106674095, 6250.966499540891, 6006.582787298896, 6204.414071586352, 5398.051421142326]

# cheetah 
margin = -100
less_nodes = [1923.5685580395225, 1855.6007820573082, 1988.0647219724683, 1995.068269461321, 1910.152494311647, 2041.2460340252524, 4385.868003433798, 1863.2874015934217, 1963.7061573039123, 1983.1297844855005]
conventional = [3149.775786965006, 1683.445476667659, 4490.643692979968, 4891.683547951092, 1816.372015481506, 4035.1772365432334, 3489.4304104512644, 2462.9374446779348, 3618.1185035811077, 1717.8521376165422]
proposed =     [4132.771766436113, 5726.321722843213, 4884.422068390368, 1905.9167686623114, 4933.837858411209, 5758.5844075421755, 5160.378457974524, 4762.131084091559, 1814.0351412470059, 4046.0185322306907]

# hopper 
margin = -57
less_nodes = [3836.0452485684677, 3893.0259766424233, 3747.3754922449066, 3788.4480238103365, 3868.07297054785, 3768.3124754226396, 3857.0749513603796, 3760.4866156625817, 3695.800034332749, 3840.725039990415]
conventional = [3864.531691138834, 3833.3619877461206, 3787.9032741449737, 3785.540661035843, 3829.219507055299, 3813.1828789777187, 3816.1289859162525, 3736.657831600011, 3835.5643879469926, 3789.5807629096016]
proposed = [3826.9881851634696, 3769.018066785692, 3767.4613364126744, 3705.4980242484717, 3830.298758406085, 3819.8048724010646, 3771.29888448375, 3690.624068458117, 3771.8896381194536, 3735.225448065652]

# pusher 
margin = -1
less_nodes = [-283.25329426864, -302.69423878542904, -294.91987592483645, -304.5164174989503, -283.8133352458524, -291.5565822512778, -287.96476277861876, -294.98044509604244, -293.1866730670407, -293.431161218066]
conventional = [-303.9340671259464, -296.9371917482907, -294.37329722715504, -291.2332905355318, -290.5464708974986, -291.77767574712396, -295.31984730039846, -286.78782245468744, -303.88475285470975, -284.8961312120762]
proposed = [-303.9340671259464, -296.9371917482907, -294.37329722715504, -291.2332905355318, -290.5464708974986, -291.77767574712396, -295.31984730039846, -286.78782245468744, -303.88475285470975, -284.8961312120762]

# reacher 
margin = -13.5
less_nodes = [-133.32521550309116, -140.2084451848645, -123.4124707248734, -121.52457405976813, -130.0228429227737, -119.26386914840495, -132.43314539175464, -122.90002099337464, -123.19119477664587, -132.18453614951923]
conventional = [-96.34354360948942, -106.95239647877591, -113.49842119252693, -116.10895576328289, -100.34133902395075, -127.06749837061425, -126.29486025649666, -112.54200529338854, -124.12851463550847, -114.34485944689398]
proposed = [-100.78196395886486, -108.71982792113417, -129.94607753309597, -127.6096578500715, -128.6431336980064, -105.27292317444706, -106.50226293109274, -129.28577723922945, -113.9471385076434, -122.90637913893025]

# swimmer 
margin = -0.2
less_nodes = [53.3596134649041, 53.19945495307631, 53.18125014491723, 52.50465922890394, 52.86184442989895, 52.51248209851578, 53.142362375583716, 53.32381320583831, 53.64658706504865, 52.59542334248854]
conventional = [53.017656147200306, 53.31875298636618, 51.33783130321011, 53.165489884412125, 52.197084704309944, 52.825101169950855, 51.77540050179813, 51.86828390535161, 52.984787595618876, 53.287112409155824]
proposed = [52.968426268326176, 53.86241847838052, 52.04385108710078, 51.780407613060916, 52.88151234606422, 53.626930697832165, 52.35139289523663, 53.70945280873051, 53.446154453773715, 52.51375682043454]

# walker 
margin = -285
less_nodes = [4269.290497868383, 3639.466421663751, 4104.224028288236, 4280.868325887414, 4092.4053489758526, 4504.717399223113, 4458.095176336722, 4058.5717680545176, 4018.3697104061785, 4461.90400473828]
conventional = [4391.081784643904, 4092.992708868545, 4493.528329002076, 4644.260401654716, 4654.789965998271, 4718.816938813808, 4022.082434679642, 4574.344458713733, 4932.137556479685, 4464.314132736429]
proposed = [4537.819704233409, 4333.224093964994, 4702.017593474808, 4414.33616338678, 4599.523392206929, 4520.732938785627, 4293.18194810131, 4443.085605637798, 3911.9966960527263, 4624.062682571202]

result = non_inferiority_test(conventional, proposed, margin=margin)
for k, v in result.items():
    print(f"{k}: {v}")
