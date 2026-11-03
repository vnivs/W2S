import torch
import os
import time
import random
import csv
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from env.fjsp_env import FJSPEnv
from env.data_generator import FJSPGenerator
from network import FJSPNetwork
from train.ppo_trainer import PPOTrainer
from utils.instance_parser import parse_fjsp_instance
from config.config_loader import load_config
from utils.proc_time_aug import get_random_augmentation

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(p: str) -> Path:
    """Resolve a path of the configuration relative to the repository."""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def create_run_directory(base_dir: str) -> str:
    """Create a run directory <base_dir>/YYYY-MM-DD-NNN."""
    today = datetime.now().strftime("%Y-%m-%d")

    run_num = 1
    while True:
        run_dir = os.path.join(base_dir, f"{today}-{run_num:03d}")
        if not os.path.exists(run_dir):
            break
        run_num += 1

    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    return run_dir


def evaluate(network, device, test_instances):
    """Greedy evaluation on the validation instances in one batch."""
    if not test_instances:
        return 0.0

    network.eval()
    bs = len(test_instances)

    first = test_instances[0]
    max_ops = max(max(job.num_operations for job in inst.jobs) for inst in test_instances)
    env = FJSPEnv({
        "n_jobs": first.n_jobs, "n_machines": first.n_machines,
        "max_ops_per_job": max_ops, "device": device,
    })

    with torch.no_grad():
        obs, _ = env.reset(options={"instances": test_instances})
        step = 0
        while not env.episode_done.all():
            encoded = _encode(env, network, device)

            if not env.can_schedule().any():
                break

            cand = env.get_candidate_scheduling_times()
            ta, _, _ = network.act_timing(encoded, cand, deterministic=True)
            sel_t = cand['times'].gather(1, ta.unsqueeze(1)).squeeze(1)

            job_op_indices = env.get_job_op_indices().to(device)
            pair_feats = env.get_pair_private_features(sel_t).to(device)
            pair_mask = env._compute_pair_action_mask(sel_t).to(device)

            pair_action, _, _ = network.act_pair(
                encoded=encoded, job_op_indices=job_op_indices,
                pair_feats=pair_feats, action_mask=pair_mask, deterministic=True)
            obs, _, _, _, _ = env.step({env.pair_agent_id: pair_action}, start_time=sel_t)
            step += 1
            if step > 10000:
                break

    network.train()
    return env.get_makespans().mean().item()


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
    """Apply the processing-time transform to the processing-time features of the pairs."""
    if aug_fn is None:
        return pair_feats
    pf = pair_feats.clone()
    pf[:, :, 6] = aug_fn(pair_feats[:, :, 6])
    pf[:, :, 8] = pf[:, :, 7] + pf[:, :, 6]
    return pf


def collect_episode(env, network, device, instances, use_proc_time_aug=True):
    """Rollout without gradients: sample (t, c) at every step and record the log-probabilities, values and makespans."""
    obs, _ = env.reset(options={"instances": instances})
    batch_size = env.batch_size
    step_count = 0
    timing_steps = 0
    n_dec = torch.zeros(batch_size, device=device)

    rec_ta, rec_logp_t, rec_v = [], [], []
    n_delay = torch.zeros(batch_size, device=device)
    n_active = torch.zeros(batch_size, device=device)
    rec_pa, rec_logp_p, rec_alive = [], [], []

    if use_proc_time_aug:
        _, aug_fn = get_random_augmentation()
    else:
        aug_fn = None

    with torch.no_grad():
        while not env.episode_done.all():
            alive = (~env.episode_done).float()
            if not env.can_schedule().any():
                break
            encoded = _encode(env, network, device, aug_fn)

            cand = env.get_candidate_scheduling_times()
            ta, logp_t, _, probs_t = network.act_timing_with_probs(
                encoded, cand, deterministic=False)

            v = network.value_state(encoded, env.get_global_features().to(device))
            active = env.can_schedule().float() * alive
            n_dec = n_dec + active
            n_active = n_active + active
            timing_steps += 1
            t_min = env.current_time
            sel_t = cand['times'].gather(1, ta.unsqueeze(1)).squeeze(1)
            n_delay = n_delay + ((sel_t > t_min) & env.can_schedule()).float() * alive

            job_op_indices = env.get_job_op_indices().to(device)
            pair_feats = _augment_pair_feats(env.get_pair_private_features(sel_t).to(device), aug_fn)
            pair_mask = env._compute_pair_action_mask(sel_t).to(device)
            pa, logp_p, _ = network.act_pair(
                encoded=encoded, job_op_indices=job_op_indices,
                pair_feats=pair_feats, action_mask=pair_mask, deterministic=False)
            n_dec = n_dec + alive

            rec_ta.append(ta); rec_logp_t.append(logp_t); rec_v.append(v)
            rec_pa.append(pa); rec_logp_p.append(logp_p); rec_alive.append(alive)

            env.step({env.pair_agent_id: pa}, start_time=sel_t)
            step_count += 1
            if step_count > 10000:
                break

    makespans = env.get_makespans().detach()
    stack = lambda xs: torch.stack(xs, dim=0) if xs else torch.zeros(0, batch_size, device=device)
    traj = {
        "timing_actions": rec_ta, "pair_actions": rec_pa,
        "old_logp_t": stack(rec_logp_t), "old_logp_p": stack(rec_logp_p),
        "v": stack(rec_v), "alive": stack(rec_alive),
        "n_delay": n_delay, "n_active": n_active,
        "n_dec": n_dec.clamp(min=1), "aug_fn": aug_fn, "T": len(rec_ta),
    }
    return makespans, traj, step_count, timing_steps


def ppo_replay_epoch(env, network, device, instances, traj, trainer,
                     adv, v_target):
    """One PPO epoch: replay the recorded actions, compute the PPO loss and accumulate the gradients."""
    obs, _ = env.reset(options={"instances": instances})
    batch_size = env.batch_size
    aug_fn = traj["aug_fn"]
    n_dec = traj["n_dec"]
    N = float(traj["n_active"].sum().clamp(min=1).item())
    old_logp_t, old_logp_p = traj["old_logp_t"], traj["old_logp_p"]
    k = 0
    agg = {"pg": 0.0, "vloss": 0.0, "ent": 0.0}
    T = traj["T"]

    while not env.episode_done.all() and k < T:
        alive = (~env.episode_done).float()
        if not env.can_schedule().any():
            break
        encoded = _encode(env, network, device, aug_fn)

        cand = env.get_candidate_scheduling_times()
        ta = traj["timing_actions"][k]
        _, logp_t, ent_t, probs_t = network.act_timing_with_probs(
            encoded, cand, deterministic=False, action=ta)

        v_pred = network.value_state(encoded, env.get_global_features().to(device))
        active = env.can_schedule().float() * alive

        sel_t = cand['times'].gather(1, ta.unsqueeze(1)).squeeze(1)

        job_op_indices = env.get_job_op_indices().to(device)
        pair_feats = _augment_pair_feats(env.get_pair_private_features(sel_t).to(device), aug_fn)
        pair_mask = env._compute_pair_action_mask(sel_t).to(device)
        pa = traj["pair_actions"][k]
        _, logp_p, ent_p = network.act_pair(
            encoded=encoded, job_op_indices=job_op_indices,
            pair_feats=pair_feats, action_mask=pair_mask, deterministic=False, action=pa)

        loss, info = trainer.step_losses(
            new_logp_t=logp_t, old_logp_t=old_logp_t[k], ent_t=ent_t,
            new_logp_p=logp_p, old_logp_p=old_logp_p[k], ent_p=ent_p,
            adv=adv[k],
            v_pred=v_pred, v_target=v_target[k], active=active,
            n_dec=n_dec, N=N)
        loss.backward()
        for key in agg:
            agg[key] += info[key]

        obs, _, dones, _, _ = env.step({env.pair_agent_id: pa}, start_time=sel_t)
        k += 1

    return agg

def train(config_path: str = "config.yaml"):
    """Training loop."""
    config = load_config(config_path)

    if config.experiment.get("seed") is not None:
        seed = config.experiment["seed"]
        torch.manual_seed(seed)
        random.seed(seed)

    device_config = config.training["device"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_config == "auto" else torch.device(device_config)

    training_sizes = config.env.get("training_sizes", [[10, 5]])
    batch_size = config.env["batch_size"]

    gen_cfg = config.get("data_gen", {})
    max_ops_per_job = config.env.get("max_ops_per_job", gen_cfg.get("ops_per_job_max", 6))

    print(f"Training sizes: {training_sizes}")
    print(f"Batch size: {batch_size} episodes")

    train_cfg = config.training
    num_iterations = train_cfg["num_iterations"]
    data_refresh_interval = train_cfg.get("data_refresh_interval", 20)
    log_interval = train_cfg["log_interval"]
    eval_interval = train_cfg.get("eval_interval", 5)
    test_data_dir = train_cfg.get("test_data_dir", "instances/song")
    save_dir = train_cfg["save_dir"]
    save_best = train_cfg["save_best"]
    save_interval = train_cfg["save_interval"]

    rl_cfg = config["optim"]
    lr = rl_cfg["lr"]
    entropy_coef = rl_cfg["entropy_coef"]
    max_grad_norm = rl_cfg["max_grad_norm"]

    ppo_cfg = config.get("ppo", {})
    ppo_clip_eps = float(ppo_cfg.get("clip_eps", 0.2))
    ppo_value_coef = float(ppo_cfg.get("value_coef", 0.5))
    ppo_gamma = float(ppo_cfg.get("gamma", 1.0))
    ppo_gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
    ppo_epochs = int(ppo_cfg.get("ppo_epochs", 4))

    timing_cfg = config.get("timing", {})
    timing_hidden_dim = timing_cfg.get("hidden_dim", 64)

    save_dir = create_run_directory(str(resolve_path(save_dir)))

    import shutil
    config_backup_path = os.path.join(save_dir, "config.yaml")
    shutil.copy(config_path, config_backup_path)

    generators = {}
    for n_jobs, n_machines in training_sizes:
        key = (n_jobs, n_machines)
        gen_kwargs = dict(
            n_jobs=n_jobs,
            n_machines=n_machines,
            ops_per_job_min=gen_cfg.get("ops_per_job_min", 3),
            ops_per_job_max=gen_cfg.get("ops_per_job_max", max_ops_per_job),
            machines_per_op_min=gen_cfg.get("machines_per_op_min", 1),
            machines_per_op_max=gen_cfg.get("machines_per_op_max", 3),
            proc_time_min=gen_cfg.get("proc_time_min", 3),
            proc_time_max=gen_cfg.get("proc_time_max", 10),
            proc_time_std=gen_cfg.get("proc_time_std", 0.5),
            output_dir=None,
            seed=None,
        )
        generators[key] = FJSPGenerator(**gen_kwargs)

    max_n_jobs = max(size[0] for size in training_sizes)
    max_n_machines = max(size[1] for size in training_sizes)

    env_config = {
        "n_jobs": max_n_jobs,
        "n_machines": max_n_machines,
        "max_ops_per_job": max_ops_per_job,
        "device": device,
    }
    env = FJSPEnv(env_config)

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
        timing_hidden_dim=timing_hidden_dim,
        use_candidate_self_attention=timing_cfg.get("use_candidate_self_attention", True),
        global_feat_dim=net_cfg.get("global_feat_dim", 2),
    ).to(device)

    trainer = PPOTrainer(
        network=network,
        lr=lr,
        clip_eps=ppo_clip_eps,
        value_coef=ppo_value_coef,
        entropy_coef=entropy_coef,
        gamma=ppo_gamma,
        gae_lambda=ppo_gae_lambda,
        max_grad_norm=max_grad_norm,
        ppo_epochs=ppo_epochs,
        device=device,
    )

    test_instances = []

    val_subdir = train_cfg.get("val_subdir", "val_1005")
    test_path = resolve_path(test_data_dir) / val_subdir

    if test_path.exists():
        test_files = sorted(test_path.glob("*.txt")) + sorted(test_path.glob("*.fjs"))
        for test_file in test_files:
            try:
                inst = parse_fjsp_instance(str(test_file))
                test_instances.append(inst)
            except Exception:
                pass
        if test_instances:
            print(f"Validation set: {val_subdir}/ ({len(test_instances)} instances)")
        else:
            print(f"No valid instances in {test_path}")
    else:
        print(f"Validation directory not found: {test_path}")

    best_makespan = float('inf')
    best_test_makespan = float('inf')

    tb_dir = os.path.join(save_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"TensorBoard logs: {tb_dir}")
    print(f"  tensorboard --logdir={tb_dir}")

    csv_path = os.path.join(save_dir, "training_log.csv")
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['iteration', 'train_makespan', 'test_makespan', 'best_test', 'loss', 'entropy', 'time'])
    csv_file.flush()


    for iteration in range(num_iterations):
        start_time = time.time()

        if iteration % data_refresh_interval == 0:
            n_jobs, n_machines = random.choice(training_sizes)
            generator = generators[(n_jobs, n_machines)]
            ep_env = FJSPEnv({
                "n_jobs": n_jobs, "n_machines": n_machines,
                "max_ops_per_job": max_ops_per_job, "device": device,
            })
            cached_instances = [generator() for _ in range(batch_size)]


        makespans, traj, avg_steps, timing_steps = collect_episode(
            ep_env, network, device, cached_instances,
            use_proc_time_aug=config.training.get("use_proc_time_aug", True))
        avg_makespan = makespans.mean().item()

        reward = trainer.standardize_reward(makespans)
        adv, v_target, _ = trainer.compute_gae(v=traj["v"], alive=traj["alive"], reward=reward)

        last = {"pg": 0.0, "vloss": 0.0, "ent": 0.0}
        grad_norm = 0.0
        for _ep in range(trainer.ppo_epochs):
            trainer.zero_grad()
            last = ppo_replay_epoch(
                ep_env, network, device, cached_instances, traj, trainer,
                adv, v_target)
            grad_norm = trainer.clip_and_step()

        _am = traj["alive"] > 0.5
        if traj["T"] > 0 and _am.any():
            v_scale = traj["v"][_am].abs().mean().item()
            tgt_scale = v_target[_am].std().item()
            adv_mean = adv[_am].mean().item()
            adv_std = adv[_am].std().item()
        else:
            v_scale = adv_mean = adv_std = 0.0
            tgt_scale = 1.0
        delay_rate = (traj["n_delay"] / traj["n_active"].clamp(min=1)).mean().item()

        metrics = {
            "policy_loss": last["pg"],
            "value_loss": last["vloss"],
            "entropy": last["ent"],
            "total_loss": last["pg"] + last["vloss"],
            "grad_norm": grad_norm,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "v_scale": v_scale,
            "tgt_scale": tgt_scale,
            "delay_rate": delay_rate,
        }

        writer.add_scalar('Train/Makespan', avg_makespan, iteration + 1)
        writer.add_scalar('Train/Steps', avg_steps, iteration + 1)
        writer.add_scalar('Loss/Total', metrics['total_loss'], iteration + 1)
        writer.add_scalar('Loss/Policy', metrics['policy_loss'], iteration + 1)
        writer.add_scalar('Loss/Value', metrics['value_loss'], iteration + 1)

        writer.add_scalar('Entropy/PerDecision', metrics['entropy'], iteration + 1)
        writer.add_scalar('Timing/Steps', timing_steps, iteration + 1)

        writer.add_scalar('Gradient/Norm', metrics['grad_norm'], iteration + 1)
        writer.add_scalar('Advantage/Mean', metrics['adv_mean'], iteration + 1)
        writer.add_scalar('Advantage/Std', metrics['adv_std'], iteration + 1)
        writer.add_scalar('Timing/DelayRate', metrics['delay_rate'], iteration + 1)
        writer.add_scalar('Critic/ValueScale', metrics['v_scale'], iteration + 1)
        writer.add_scalar('Critic/TargetScale', metrics['tgt_scale'], iteration + 1)

        test_makespan_val = None
        if test_instances and (iteration + 1) % eval_interval == 0:
            test_makespan_val = evaluate(
                network=network, device=device, test_instances=test_instances,
            )
            if save_best and test_makespan_val < best_test_makespan:
                best_test_makespan = test_makespan_val
            writer.add_scalar('Test/Makespan', test_makespan_val, iteration + 1)
            writer.add_scalar('Test/Best_Makespan', best_test_makespan, iteration + 1)

        if avg_makespan < best_makespan:
            best_makespan = avg_makespan

        if (iteration + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            print(f"Iter {iteration + 1}/{num_iterations} | "
                  f"MS: {avg_makespan:.2f} (best: {best_makespan:.2f}) | "
                  f"Steps: {avg_steps} | TSteps: {timing_steps} | "
                  f"Loss: {metrics['total_loss']:.4f} (P {metrics['policy_loss']:.4f} "
                  f"V {metrics['value_loss']:.4f}) | "
                  f"Ent: {metrics['entropy']:.3f} | "
                  f"delay: {metrics['delay_rate']*100:.1f}% | "
                  f"|V|{metrics['v_scale']:.3f}/tgt{metrics['tgt_scale']:.2f} | "
                  f"GNorm: {metrics['grad_norm']:.3f} | "
                  f"Time: {elapsed:.2f}s", flush=True)

        if test_makespan_val is not None:
            print(f"\nTest MS: {test_makespan_val:.2f} (best: {best_test_makespan:.2f})", flush=True)

            if save_best and test_makespan_val <= best_test_makespan:
                save_path = os.path.join(save_dir, "best_test_model.pt")
                torch.save({"network_state_dict": network.state_dict()}, save_path)
                print(f"Saved the best validation model (makespan {best_test_makespan:.2f})")

        elapsed = time.time() - start_time
        csv_writer.writerow([
            iteration + 1,
            f"{avg_makespan:.4f}",
            f"{test_makespan_val:.4f}" if test_makespan_val else "",
            f"{best_test_makespan:.4f}" if best_test_makespan < float('inf') else "",
            f"{metrics['total_loss']:.6f}",
            f"{metrics['entropy']:.4f}",
            f"{elapsed:.2f}"
        ])
        csv_file.flush()

        if save_interval > 0 and (iteration + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f"model_iter_{iteration + 1}.pt")
            torch.save({"network_state_dict": network.state_dict()}, save_path)

    csv_file.close()
    writer.close()

if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    train(config_path=config_path)
