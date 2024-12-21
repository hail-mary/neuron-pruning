import os
import yaml
import pprint
import cloudpickle


class Logger:
    def __init__(self, cfg):     
        self.iteration = 0
        # make log directory if not exists
        if not os.path.exists(cfg['logdir']):
            os.mkdir(cfg['logdir'])

         # Pretty-print the configuration
        print('\n------------------ Loaded Configuration --------------------')
        pprint.pprint(cfg)

        # save configuration
        config_dir = os.path.join(cfg['logdir'], 'config.yaml')
        with open(config_dir, 'w') as file:
            yaml.dump(cfg, file, indent=4)
            print(f'\n >> saved to {config_dir}.')

        self.cfg = cfg
    
    def step(self):
        self.iteration += 1
    
    def save_checkpoint(self, results, save_to='checkpoints'):
        # results = [[worker_id, reward, policy_kwargs, policy_weight],
        #              worker_id, reward, policy_kwargs, policy_weight], ... ]

        # sort results in reward decending order
        results.sort(key=lambda x: x[1], reverse=True)

        # find the best worker and save at every checkpoints
        best_worker = results[0][0]
        best_reward = results[best_worker][1]
        best_policy_kwargs = results[best_worker][2]
        best_policy_weight = results[best_worker][3]

        idx = self.iteration
        checkpoints_dir = os.path.join(self.cfg['logdir'], save_to)
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        path_to_model = os.path.join(checkpoints_dir, f'Iteration-{idx}')
        os.mkdir(path_to_model)
        
        with open(f'{path_to_model}/best_policy_kwargs.pkl', 'wb') as f:
            cloudpickle.dump(best_policy_kwargs, f)
        with open(f'{path_to_model}/best_policy_weights.pkl', 'wb') as f:
            cloudpickle.dump(best_policy_weight, f)

       