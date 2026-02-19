def estimate_flops_and_speedup(
    input_dim: int,
    output_dim: int,
    hidden_layers: list,
    total_epochs: int,
    epoch_step: int,
    decay_factors: list
):
    """
    入出力次元と初期構造を元に、縮小スケジュールに基づく訓練FLOPsと相対削減量を見積もる関数。
    
    Parameters:
        input_dim (int): 入力次元数
        output_dim (int): 出力次元数
        hidden_layers (list): 隠れ層の初期構造 (例：[256, 256])
        total_epochs (int): 総エポック数 (例：1000)
        epoch_step (int): 構造を縮小する間隔のエポック数 (例：250)
        decay_factors (list): 各層に対する縮小係数（例：[0.7, 0.6]）

    Returns:
        dict: FLOPsの集計と短縮割合
    """

    def compute_flops(layers):
        """層構造に基づいて1データあたりのFLOPsを計算"""
        flops = input_dim * layers[0]
        for i in range(len(layers) - 1):
            flops += layers[i] * layers[i + 1]
        flops += layers[-1] * output_dim
        return flops

    num_steps = total_epochs // epoch_step
    current_layers = list(hidden_layers)
    
    flops_total_reduced = 0
    flops_total_full = compute_flops(hidden_layers) * total_epochs

    print(f"--- 各ステップごとの構造とFLOPs ---")
    for step in range(num_steps):
        flops_step = compute_flops(current_layers)
        flops_total_reduced += flops_step * epoch_step
        print(f"Epochs {step*epoch_step+1}-{(step+1)*epoch_step}: {current_layers} -> {flops_step} FLOPs/ep")

        # 次のステップで構造を縮小
        current_layers = [
            # max(1, int(current_layers[i] * decay_factors[i]))
            max(1, current_layers[i] - int(current_layers[i] * decay_factors[i]))
            for i in range(len(current_layers))
        ]

    reduction_ratio = 1 - (flops_total_reduced / flops_total_full)

    print(f"\n✅ 総FLOPs（一定構造）: {flops_total_full:,}")
    print(f"✅ 総FLOPs（縮小構造）: {flops_total_reduced:,}")
    print(f"📉 推定訓練時間短縮: {reduction_ratio * 100:.2f}%")

    return {
        "flops_full": flops_total_full,
        "flops_reduced": flops_total_reduced,
        "reduction_ratio": reduction_ratio
    }


# ✅ 例：256-256構造、入力100、出力10、250エポックごとに縮小（1000エポック）
estimate_flops_and_speedup(
    input_dim=105,
    output_dim=8,
    hidden_layers=[256, 256],
    total_epochs=1000,
    epoch_step=250,
    decay_factors=[0.3, 0.4]
)
