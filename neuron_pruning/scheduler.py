import torch
import numpy as np

class Scheduler:
    """
    a set of utility functions for training
    """
    def __init__(self, cfg):
        self.cfg = cfg

    
    def average_params(self, params_list):
        avg_params = {}
        for key in params_list[0].keys():
            avg_params[key] = torch.stack([params[key] for params in params_list]).mean(dim=0)
        return avg_params
    
    def calculate_norms(self, pre_avg_params, avg_params):
        pre_avg_norm = torch.cat([p.flatten() for p in pre_avg_params.values()]).norm()
        avg_norm = torch.cat([p.flatten() for p in avg_params.values()]).norm()
        return pre_avg_norm, avg_norm

    def preprocess(self, raw_arch, raw_params):
        # First, preprocess arch
        arch = {
            'policy': raw_arch['pi'].copy(),
            'value': raw_arch['vf'].copy() if 'vf' in raw_arch else raw_arch.get('qf', []).copy()
        }

        # Then, preprocess params
        params = {}
        for layer_type, layers in arch.items():
            for key, val in raw_params.items():
                if layer_type in key and 'weight' in key:
                    params.update({key: val})
                elif layer_type in key and 'bias' in key:
                    params.update({key: val})
                elif 'action' in key and layer_type == 'policy':
                    params.update({key: val})

        # keep auxilary info to reconstruct
        aux = {'log_std': raw_params.pop('log_std'), 
               'action_net.bias': raw_params.pop('action_net.bias'),
               'value_net.bias': raw_params.pop('value_net.bias'),
        }

        return arch, params, aux

    def reconstruct(self, arch, params, aux):
        # First, reconstruct arch
        processed_arch = {
            'pi': arch['policy'],
            'vf': arch['value']
        }

        # Then, reconstruct params
        # Missing key(s) in state_dict: "log_std", "action_net.bias", "value_net.bias"
        processed_params = params
        processed_params.update(aux)

        return processed_arch, processed_params

    def remove_indices(self, tensor, indices_to_remove, row_or_col=None):
        if tensor.dim() == 1: # for bias
            # Create a boolean mask indicating which elements to keep
            mask = torch.ones(tensor.shape[0], dtype=torch.bool)
            mask[indices_to_remove] = False
        elif tensor.dim() == 2 and row_or_col == 'row': # for weight
            # Create a boolean mask indicating which rows to keep
            mask = torch.ones_like(tensor, dtype=torch.bool)
            mask[indices_to_remove, :] = False
        elif tensor.dim() == 2 and row_or_col == 'col': # for weight
            # Create a boolean mask indicating which columns to keep
            mask = torch.ones_like(tensor, dtype=torch.bool)
            mask[:, indices_to_remove] = False
        else:
            raise ValueError("Invalid tensor dimension or row_or_col value")

        # Apply the mask to select the desired elements
        new_tensor = tensor[mask]
        if row_or_col == 'row':
            new_units = tensor.shape[0] - len(indices_to_remove)
            new_tensor = new_tensor.reshape(new_units, tensor.shape[1])
        elif row_or_col == 'col':
            new_units = tensor.shape[1] - len(indices_to_remove)
            new_tensor = new_tensor.reshape(tensor.shape[0], new_units)

        return new_tensor

    def get_num_to_drop(self, iteration, target_sparsity, arch):
        # dropout rates are considered as the final sparsity level
        T_end = self.cfg['T_end']
        num_to_drop = {}
        
        # Get original architecture for reference
        raw_orig_arch = self.cfg['policy_kwargs']['net_arch']
        # Map it using the same logic as in preprocess ('pi' -> 'policy', 'vf' -> 'value')
        orig_arch = {
            'policy': raw_orig_arch['pi'],
            'value': raw_orig_arch['vf'] if 'vf' in raw_orig_arch else raw_orig_arch.get('qf', [])
        }

        for layer_type, target_rates in target_sparsity.items():
            num_to_drop[layer_type] = []
            for layer_idx in range(len(target_rates)):
                target_s = target_rates[layer_idx]
                
                # Pruning schedule: s(t) = target_s * (1 - (1 - t/T_end)^3)
                progress = min(1.0, iteration / T_end)
                current_target_s = target_s * (1 - (1 - progress) ** 3)
                
                orig_count = orig_arch[layer_type][layer_idx]
                current_count = arch[layer_type][layer_idx]
                
                # Target number of neurons to remain
                target_count = int(orig_count * (1 - current_target_s))
                # How many to drop in this step
                drop_count = current_count - target_count
                
                num_to_drop[layer_type].append(max(0, drop_count))
        return num_to_drop

    def modify_network(self, params, arch, iteration, target_sparsity):
        num_to_drop_dict = self.get_num_to_drop(iteration, target_sparsity, arch)
        for layer_type, layers in arch.items():
            for layer_idx in range(len(layers)):
                n_drop = num_to_drop_dict[layer_type][layer_idx]
                if n_drop > 0:
                    print(f"Layer {layer_type} {layer_idx}: {layers[layer_idx]} neurons, {n_drop} neurons to drop")
                    # num_dropped = 256 - layers[layer_idx]
                    # num_to_drop = int(256 * current_sparsity - num_dropped)
                    # First, modify network architecture
                    arch[layer_type][layer_idx] -= n_drop

                    # Then, modify network parameters
                    # for random elimination:
                    # indices_to_remove = np.random.choice(layers[layer_idx], num_to_drop, replace=False)

                    # Process weights
                    weight_key_1 = f"mlp_extractor.{layer_type}_net.{2 * layer_idx}.weight"  # Adjust the key format as per your architecture
                    if weight_key_1 in params:
                        weight_magnitude = torch.sqrt(torch.sum(params[weight_key_1] ** 2, dim=1))
                        values, indices_to_remove = torch.topk(weight_magnitude, n_drop, largest=False)
                        params[weight_key_1] = self.remove_indices(params[weight_key_1], indices_to_remove, row_or_col='row')
                    
                    # Process biases
                    bias_key = f"mlp_extractor.{layer_type}_net.{2 * layer_idx}.bias"  # Adjust the key format as per your architecture
                    if bias_key in params:
                        params[bias_key] = self.remove_indices(params[bias_key], indices_to_remove)

                    # Process weights
                    weight_key_2 = f"mlp_extractor.{layer_type}_net.{2 * (layer_idx + 1)}.weight"  # Adjust the key format as per your architecture
                    if weight_key_2 in params:
                        params[weight_key_2] = self.remove_indices(params[weight_key_2], indices_to_remove, row_or_col='col')

                    # Process output
                    output_key = "action_net.weight" if layer_type == 'policy' else "value_net.weight"
                    if output_key in params and layer_idx == len(layers) - 1 :
                        params[output_key] = self.remove_indices(params[output_key], indices_to_remove, row_or_col='col')

        return arch, params

