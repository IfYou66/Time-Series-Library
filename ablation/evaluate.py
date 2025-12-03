import argparse
import time

import numpy as np
import torch
from thop import profile

from ablation.TimesNetAdv import Model


DEFAULT_CONFIG = {
    # 任务/训练相关（保持与 run.py 一致的接口，方便直接复用命令）
    "task_name": "classification",
    "is_training": 0,
    "model_id": "Hell",
    "model": "TimesNetAdv",
    "des": "dual_learnable_7k",
    "itr": 1,
    "learning_rate": 1e-3,
    "train_epochs": 30,
    "patience": 10,
    "num_workers": 0,
    "root_path": "./dataset/damage_detection_hell/",
    "data": "Hell",
    # 数据/模型尺寸
    "seq_len": 96,
    "pred_len": 0,
    "enc_in": 40,
    "num_class": 10,
    "d_model": 16,
    "d_ff": 32,
    "dropout": 0.1,
    "e_layers": 2,
    "batch_size": 16,
    # TimesNetAdv 结构参数
    "top_k": 3,
    "moving_avg": 25,
    "min_period": 3,
    "conv2d_dropout": 0.1,
    "cyc_conv_kernel": 9,
    "sk_tau": 1.5,
    "se_strength": 0.0,
    "use_series_decomp": 1,
    "use_sk": 1,
    "use_se": 0,
    "use_cyc_conv1d": 1,
    "use_gate_mlp": 1,
    "use_res_scale": 1,
    "reflect_pad": 1,
    "use_dual_path": 1,
    "local_kernel": 7,
    "use_global_attn": 0,
    "dual_fusion_mode": "learnable",
    "gate_alpha": 5.0,
    "gate_tau": 0.45,
    "dual_beta": 0.25,
    "gate_warmup": 0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark TimesNetAdv with classification configs.")
    for key, value in DEFAULT_CONFIG.items():
        arg_type = type(value)
        if arg_type is bool:
            arg_type = int  # 与原命令行保持 0/1 风格
        parser.add_argument(f"--{key}", type=arg_type, default=value)
    parser.add_argument("--print_config", action="store_true",
                        help="打印最终配置（可选，用于确认参数）")
    return parser


class Configs:
    """简单容器，把 argparse.Namespace/字典转成属性，兼容训练脚本的 Model 接口。"""

    def __init__(self, args: argparse.Namespace):
        merged = DEFAULT_CONFIG.copy()
        merged.update(vars(args))
        for key, value in merged.items():
            setattr(self, key, value)


# ==========================================
# 测速主函数
# ==========================================
def run_benchmark(parsed_args: argparse.Namespace):
    # 0. 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    configs = Configs(parsed_args)
    if parsed_args.print_config:
        print("---- Effective Config ----")
        for key in sorted(vars(configs).keys()):
            print(f"{key}: {getattr(configs, key)}")
        print("--------------------------")

    # 实例化模型
    model = Model(configs).to(device)
    model.eval()

    # 构造假输入 [Batch, Seq_Len, Channels]
    batch_size = configs.batch_size
    dummy_input = torch.randn(batch_size, configs.seq_len, configs.enc_in, device=device)

    print("-" * 50)
    print(f"Model: TimesNetAdv (Dual-Path={'ON' if configs.use_dual_path else 'OFF'})")
    print("-" * 50)

    # ==========================================
    # 指标 1: 参数量 (Parameters)
    # ==========================================
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"1. Parameters (参数量): {params / 1e6:.3f} M")

    # ==========================================
    # 指标 2: 理论计算量 (FLOPs)
    # ==========================================
    print("2. Calculating FLOPs...")
    try:
        input_for_flops = torch.randn(1, configs.seq_len, configs.enc_in, device=device)
        macs, _ = profile(model, inputs=(input_for_flops,), verbose=False)
        # 1 MAC (乘加运算) ≈ 2 FLOPs，但学术界通常直接汇报 MACs 或将其称为 FLOPs
        flops_g = macs / 1e9
        print(f"   Theoretical FLOPs: {flops_g:.3f} G (Batch Size=1)")
    except Exception as e:
        print(f"   FLOPs calculation failed: {e}")

    # ==========================================
    # 指标 3: 实际推理延迟 (Inference Latency)
    # ==========================================
    print("3. Measuring Latency (推理延迟)...")

    # 3.1 预热 (Warm-up) - 非常重要！
    # GPU 在刚开始运行时需要初始化 context，速度会慢，必须预热
    print("   -> Warming up model...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 3.2 正式测量
    repetitions = 100  # 重复跑100次取平均
    timings = []

    with torch.no_grad():
        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for _ in range(repetitions):
                start_event.record()
                _ = model(dummy_input)
                end_event.record()
                torch.cuda.synchronize()
                curr_time = start_event.elapsed_time(end_event)  # 毫秒
                timings.append(curr_time)
        else:
            for _ in range(repetitions):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0)  # 转毫秒

    avg_latency = np.mean(timings)
    std_latency = np.std(timings)
    throughput = (batch_size * 1000) / avg_latency

    print(f"   Avg Latency: {avg_latency:.2f} ms ± {std_latency:.2f} ms (Batch={batch_size})")
    print(f"   Throughput : {int(throughput)} samples/sec")
    print("-" * 50)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_benchmark(args)
