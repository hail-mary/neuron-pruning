import torch
import numpy as np

class Trainer:
    """
    a set of utility functions for training
    """
    def __init__(self, cfg):
        self.cfg = cfg

    
    def average_weights(self, weights_list):
        avg_weights = {}
        for key in weights_list[0].keys():
            avg_weights[key] = torch.stack([weights[key] for weights in weights_list]).mean(dim=0)
        return avg_weights

    def preprocess(self, raw_arch, raw_params):
        # First, preprocess arch
        layer_types = ('policy', 'value')
        archs = tuple(raw_arch.values())
        arch = dict(zip(layer_types, archs))

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
        layer_types = ('pi', 'vf')
        archs = tuple(arch.values())
        processed_arch = dict(zip(layer_types, archs))

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

    def modify_network(self, params, arch, dropout_rates):
        for layer_type, layers in arch.items():
            for layer_idx in range(len(layers)):
                if dropout_rates[layer_type][layer_idx] > 0.0:
                    num_to_drop = int(layers[layer_idx] * dropout_rates[layer_type][layer_idx])
                    if num_to_drop == 0:
                        continue

                    # First, modify network architecture
                    arch[layer_type][layer_idx] -= num_to_drop

                    # Then, modify network parameters
                    indices_to_remove = np.random.choice(layers[layer_idx], num_to_drop, replace=False)
                    # print(layer_type, layer_idx, len(indices_to_remove))

                    # Process weights
                    weight_key_1 = f"mlp_extractor.{layer_type}_net.{2 * layer_idx}.weight"  # Adjust the key format as per your architecture
                    if weight_key_1 in params:
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

