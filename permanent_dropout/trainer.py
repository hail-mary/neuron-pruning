import torch


layer_types = ('policy', 'value')
archs = tuple(model.policy_kwargs['net_arch'].values())
weights = model.policy.state_dict()
arch = dict(zip(layer_types, archs))

class Trainer:
    """
    a set of utility functions for training
    """
    def __init__(self, cfg):
        self.cfg = cfg

    
    def average_weights(self, weights_list):
        """
        重みを平均化する。
        """
        avg_weights = {}
        for key in weights_list[0].keys():
            avg_weights[key] = torch.stack([weights[key] for weights in weights_list]).mean(dim=0)
        return avg_weights

    def reshape_params(self, weights):
        params = {}
        for layer_type, layers in arch.items():
            for key, val in weights.items():
                if layer_type in key and 'weight' in key:
                    params.update({key: val})
                elif layer_type in key and 'bias' in key:
                    params.update({key: val})
                elif 'action' in key and layer_type == 'policy':
                    params.update({key: val})
        params.pop('action_net.bias')
        params.pop('value_net.bias')

        return params

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
    
    
    def modify_network(self, weights, arch, dropout_rates, device):
        """
        ネットワークのアーキテクチャを修正し、指定割合のニューロンを削除。
        """
        new_arch = {}
        modified_weights = {}

        for layer_type, layers in arch.items():  # layer_type は 'pi' または 'vf'
            new_arch[layer_type] = []
            for idx, units in enumerate(layers):
                # 各層のドロップアウト率を取得
                dropout_rate = dropout_rates[layer_type][idx]
                new_units = int(units * (1 - dropout_rate))  # 削除後のユニット数
                new_arch[layer_type].append(new_units)

            # 重みを修正
            for key, value in weights.items():
                if layer_type in key and "weight" in key:
                    # 各層の削除されたニューロンに対応する重みを除外
                    layer_idx = int(key.split(".")[1])  # 層のインデックスを取得
                    if layer_idx < len(layers):
                        new_units = new_arch[layer_type][layer_idx]
                        modified_weights[key] = value[:new_units, :new_units].to(device)
                    else:
                        modified_weights[key] = value.to(device)
                else:
                    modified_weights[key] = value.to(device)

        return new_arch, modified_weights