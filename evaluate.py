"""Inference with a trained W2S model.

  -G  greedy decoding                          --mode greedy --no_augment
  -A  greedy decoding on 8 transformed copies    --mode greedy (default)
  -Sk sampling of k solutions                    --mode sampling --num_samples k --no_augment

The transforms change only the processing times seen by the network; every schedule is built and evaluated with the
true processing times, and the best makespan over the copies or samples is reported.
"""
import torch
import argparse
import time
from pathlib import Path

from env.fjsp_env import FJSPEnv
from network.fjsp_network import FJSPNetwork
from utils.instance_parser import parse_fjsp_instance
from config.config_loader import load_config
from utils.proc_time_aug import get_all_augmentations, vectorized_augment_all, vectorized_augment_batch, AUGMENTATIONS


def _encode(env, network, device, aug_fn=None):
    """Encode the current state."""
    net_input = env.get_network_input()
    proc_time = net_input["proc_time"].to(device)
    if aug_fn is not None:
        proc_time = aug_fn(proc_time)
    return network.encode(
        ope_feats=net_input["ope_feats"].to(device),
        ma_feats=net_input["ma_feats"].to(device),
        proc_time=proc_time,
        ope_ma_adj=net_input["ope_ma_adj"].to(device),
        ope_pre_adj=net_input["ope_pre_adj"].to(device),
        ope_sub_adj=net_input["ope_sub_adj"].to(device),
        ope_valid_mask=net_input["ope_valid_mask"].to(device),
    )


def _augment_pair_feats(pair_feats, aug_fn):
    """Apply a processing-time transform to the processing-time features of the pairs."""
    if aug_fn is None:
        return pair_feats
    pf = pair_feats.clone()
    pf[:, :, 6] = aug_fn(pair_feats[:, :, 6])
    pf[:, :, 8] = pf[:, :, 7] + pf[:, :, 6]
    return pf


def _encode_batched_aug(env, network, device, aug_list):
    """Encode the state with a different processing-time transform for every batch element."""
    net_input = env.get_network_input()
    proc_time_raw = net_input["proc_time"].to(device)
    if len(aug_list) == len(AUGMENTATIONS):
        proc_time = vectorized_augment_all(proc_time_raw[0])
    else:
        proc_time = proc_time_raw.clone().float()
        for i, (_, aug_fn) in enumerate(aug_list):
            proc_time[i:i+1] = aug_fn(proc_time_raw[i:i+1])
    return network.encode(
        ope_feats=net_input["ope_feats"].to(device),
        ma_feats=net_input["ma_feats"].to(device),
        proc_time=proc_time,
        ope_ma_adj=net_input["ope_ma_adj"].to(device),
        ope_pre_adj=net_input["ope_pre_adj"].to(device),
        ope_sub_adj=net_input["ope_sub_adj"].to(device),
        ope_valid_mask=net_input["ope_valid_mask"].to(device),
    )


def _batch_augment_pair_feats(pair_feats, aug_list):
    """Apply a different processing-time transform to the pair features of every batch element."""
    pf = pair_feats.clone()
    if len(aug_list) == len(AUGMENTATIONS):
        pf[:, :, 6] = vectorized_augment_batch(pair_feats[:, :, 6])
    else:
        for i, (_, aug_fn) in enumerate(aug_list):
            pf[i:i+1, :, 6] = aug_fn(pair_feats[i:i+1, :, 6])
    pf[:, :, 8] = pf[:, :, 7] + pf[:, :, 6]
    return pf


def _collect_sampling_batch(env, network, device, instance, num_samples, aug_fn=None):
    """Batched sampling rollouts; returns the makespans of all samples."""
    instances_batch = [instance] * num_samples
    obs, _ = env.reset(options={"instances": instances_batch})

    network.eval()
    with torch.no_grad():
        step = 0
        while not env.episode_done.all():
            encoded = _encode(env, network, device, aug_fn)

            if not env.can_schedule().any():
                break

            cand = env.get_candidate_scheduling_times()
            ta, _, _ = network.act_timing(encoded, cand, deterministic=False)
            sel_t = cand['times'].gather(1, ta.unsqueeze(1)).squeeze(1)

            job_op_indices = env.get_job_op_indices().to(device)
            pair_feats = _augment_pair_feats(env.get_pair_private_features(sel_t).to(device), aug_fn)
            action_mask = env._compute_pair_action_mask(sel_t).to(device)
            pair_action, _, _ = network.act_pair(
                encoded=encoded, job_op_indices=job_op_indices,
                pair_feats=pair_feats, action_mask=action_mask, deterministic=False,
            )
            obs, _, _, _, _ = env.step({env.pair_agent_id: pair_action}, start_time=sel_t)
            step += 1
            if step > 10000:
                break

    return env.get_makespans().cpu().tolist()


def evaluate_single_instance(network, instance_path, device, mode='greedy', num_samples=10, use_augment=True):
    """Solve one instance; returns (best makespan, wall-clock time in s)."""
    start_time = time.time()

    instance = parse_fjsp_instance(instance_path)
    max_ops_per_job = max(job.num_operations for job in instance.jobs)
    env_cfg = {
        "n_jobs": instance.n_jobs,
        "n_machines": instance.n_machines,
        "max_ops_per_job": max_ops_per_job,
        "device": device,
    }

    from utils.proc_time_aug import aug_identity
    aug_list = get_all_augmentations() if use_augment else [("identity", aug_identity)]
    n_aug = len(aug_list)

    env = FJSPEnv(env_cfg)
    all_makespans = []

    if mode == 'greedy':
        # One copy of the instance per transform, all decoded greedily in one batch
        network.eval()
        obs, _ = env.reset(options={"instances": [instance] * n_aug})
        with torch.no_grad():
            step = 0
            while not env.episode_done.all():
                encoded = _encode_batched_aug(env, network, device, aug_list)
                if not env.can_schedule().any():
                    break

                cand = env.get_candidate_scheduling_times()
                ta, _, _ = network.act_timing(encoded, cand, deterministic=True)
                sel_t = cand['times'].gather(1, ta.unsqueeze(1)).squeeze(1)

                job_op_indices = env.get_job_op_indices().to(device)
                pair_feats = _batch_augment_pair_feats(env.get_pair_private_features(sel_t).to(device), aug_list)
                action_mask = env._compute_pair_action_mask(sel_t).to(device)
                pair_action, _, _ = network.act_pair(
                    encoded=encoded, job_op_indices=job_op_indices,
                    pair_feats=pair_feats, action_mask=action_mask, deterministic=True)
                obs, _, _, _, _ = env.step({env.pair_agent_id: pair_action}, start_time=sel_t)
                step += 1
                if step > 10000:
                    break
        all_makespans = env.get_makespans().cpu().tolist()

    elif mode == 'sampling':
        # num_samples sampled rollouts per transform
        for _, aug_fn in aug_list:
            ms_list = _collect_sampling_batch(env, network, device, instance, num_samples, aug_fn=aug_fn)
            all_makespans.extend(ms_list)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'greedy' or 'sampling'.")

    inference_time = time.time() - start_time
    return min(all_makespans), inference_time


def evaluate_benchmark(network, benchmark_dir, device, mode='greedy', num_samples=10, use_augment=True):
    """Evaluate all instances of a benchmark directory."""
    benchmark_path = Path(benchmark_dir)
    instance_files = (sorted(benchmark_path.glob("**/*.txt")) + sorted(benchmark_path.glob("**/*.fjs"))
                      + sorted(benchmark_path.glob("**/*.jsp")))

    results = {}
    total_time = 0.0

    for inst_file in instance_files:
        try:
            makespan, inference_time = evaluate_single_instance(network, str(inst_file), device, mode,
                                                                num_samples, use_augment=use_augment)
            results[inst_file.stem] = makespan
            total_time += inference_time
            print(f"{inst_file.stem}: {makespan:.2f} (time: {inference_time:.2f}s)")
        except Exception as e:
            print(f"{inst_file.stem}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            results[inst_file.stem] = None

    if results:
        print(f"\n{'='*50}")
        print(f"Total inference time: {total_time:.2f}s")
        valid_count = sum(1 for v in results.values() if v is not None)
        if valid_count > 0:
            print(f"Average time per instance: {total_time/valid_count:.2f}s")
        print(f"{'='*50}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained W2S model")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to the checkpoint")
    parser.add_argument("--benchmark", type=str, default="instances/brandimarte", help="directory of the instances")
    parser.add_argument("--config", type=str, default="config.yaml", help="configuration file")
    parser.add_argument("--mode", type=str, default="greedy", choices=["greedy", "sampling"],
                        help="greedy (-G, or -A with the augmentation) or sampling (-S)")
    parser.add_argument("--num_samples", type=int, default=10, help="number of samples in sampling mode")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda or cpu")
    parser.add_argument("--no_augment", action="store_true", help="disable the processing-time augmentation (enabled by default)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    config = load_config(args.config)
    net_cfg = config.network

    network = FJSPNetwork(
        in_size_ope=net_cfg["in_size_ope"],
        in_size_ma=net_cfg["in_size_ma"],
        pair_private_dim=net_cfg["pair_private_dim"],
        out_size_ope=net_cfg["out_size_ope"],
        out_size_ma=net_cfg["out_size_ma"],
        hidden_size=net_cfg["hidden_size"],
        num_hgnn_layers=net_cfg["num_hgnn_layers"],
        num_heads=net_cfg["num_heads"],
        actor_hidden_dim=net_cfg["actor_hidden_dim"],
        actor_num_layers=net_cfg["actor_num_layers"],
        dropout=net_cfg["dropout"],
        pair_attn_layers=net_cfg.get("pair_attn_layers", 2),
        timing_scalar_dim=9,
        timing_hidden_dim=config.get("timing", {}).get("hidden_dim", 64),
        use_candidate_self_attention=config.get("timing", {}).get("use_candidate_self_attention", True),
        global_feat_dim=net_cfg.get("global_feat_dim", 2),
    ).to(device)

    print(f"Checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # The released checkpoints do not contain the critic, which is not needed for inference
    missing, unexpected = network.load_state_dict(checkpoint["network_state_dict"], strict=False)
    assert not unexpected and all(k.startswith("critic.") for k in missing), (missing, unexpected)

    print(f"Instances: {args.benchmark}")
    print(f"Mode: {args.mode} | augmentation: {'on (8 transforms)' if not args.no_augment else 'off'}")
    if args.mode == 'sampling':
        print(f"Samples: {args.num_samples}")
    print("-" * 40)

    results = evaluate_benchmark(network, args.benchmark, device, args.mode,
                                  args.num_samples, use_augment=not args.no_augment)

    print("-" * 40)
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        avg_makespan = sum(valid_results.values()) / len(valid_results)
        print(f"Mean makespan: {avg_makespan:.2f}")
        print(f"Instances solved: {len(valid_results)}/{len(results)}")

    return results


if __name__ == "__main__":
    main()
