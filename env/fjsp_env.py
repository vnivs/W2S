import torch
import numpy as np
from typing import Dict, List, Optional
from utils.instance_parser import FJSPInstance, parse_fjsp_instance


class FJSPEnv:
    """Batched FJSP environment with the factorized timing–pairing action; all instances of a batch have the same size.

    Operations are indexed job by job (index = job * max_ops_per_job + op) and padded to max_ops_per_job per job.
    A step appends the selected operation to the end of its machine sequence (Section 4.1 of the paper):
      1. the timing head picks t from get_candidate_scheduling_times()  (timing action space, Eq. (4));
      2. the pairing head picks a (job, machine) pair whose earliest start time equals t  (Eq. (6));
      3. step() starts the operation at t on that machine.
    The episode ends after every operation has been scheduled; the only reward is -C_max at the last step.
    """

    def __init__(self, config: dict):
        self.n_jobs = config.get("n_jobs")
        self.n_machines = config.get("n_machines")
        self.max_ops_per_job = config.get("max_ops_per_job")
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        self.pair_agent_id = "pair_agent"

        # Static instance data (set in reset)
        self.batch_size = None
        self.job_ops_count = None
        self.op_machines_orig = None
        self.op_proc_times_orig = None
        self.total_operations = None
        self.max_time_estimate = None

        # Graph structure: job of every operation, predecessor / successor adjacency, valid (non-padded) operations
        self.opes_appertain = None
        self.ope_pre_adj = None
        self.ope_sub_adj = None
        self.nums_opes = None
        self.ope_valid_mask = None

        # Raw node features and operation–machine edges seen by the encoder
        self.feat_opes = None
        self.feat_mas = None
        self.proc_time = None
        self.ope_ma_adj = None

        # Dynamic state: next operation of every job, available times T^J (jobs) and T^M (machines)
        self.job_current_op = None
        self.machine_available_time = None
        self.machine_utiliz_time = None
        self.job_available_time = None
        self.job_cumulative_proc_time = None
        self.current_time = None
        self.scheduled_operations = None
        self.makespan = None

        self._pair_start_cache = None

        self.episode_done = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple:
        """Load a batch of instances (options['instances'] or options['instance_path']) and return the initial observation."""
        if seed is not None:
            torch.manual_seed(seed)

        if options and "instances" in options:
            instances = options["instances"]
        elif options and "instance_path" in options:
            instances = [parse_fjsp_instance(options["instance_path"])]
        else:
            raise ValueError("either 'instances' or 'instance_path' must be given")

        self.batch_size = len(instances)
        self._pair_start_cache = None
        self._validate_instances(instances)
        self._preprocess_instances(instances)
        self._init_state()
        self._init_features()

        obs_batch = self._get_pair_observations()
        return {self.pair_agent_id: obs_batch}, {self.pair_agent_id: {}}

    def _validate_instances(self, instances: List[FJSPInstance]):
        for i, inst in enumerate(instances):
            if inst.n_jobs != self.n_jobs:
                raise ValueError(f"instance {i}: n_jobs={inst.n_jobs} != {self.n_jobs}")
            if inst.n_machines != self.n_machines:
                raise ValueError(f"instance {i}: n_machines={inst.n_machines} != {self.n_machines}")

    def _preprocess_instances(self, instances: List[FJSPInstance]):
        """Convert the instances to batched tensors."""
        batch_size = self.batch_size
        n_jobs = self.n_jobs
        n_machines = self.n_machines
        max_ops = self.max_ops_per_job
        n_ops = n_jobs * max_ops

        job_ops_count_np = np.zeros((batch_size, n_jobs), dtype=np.int64)
        op_machines_np = np.zeros((batch_size, n_jobs, max_ops, n_machines), dtype=np.bool_)
        op_proc_times_np = np.zeros((batch_size, n_jobs, max_ops, n_machines), dtype=np.int64)

        for b, instance in enumerate(instances):
            for job_id, job in enumerate(instance.jobs):
                job_ops_count_np[b, job_id] = job.num_operations
                for op_id, op in enumerate(job.operations):
                    for m_id in op.available_machines:
                        op_machines_np[b, job_id, op_id, m_id] = True
                        op_proc_times_np[b, job_id, op_id, m_id] = op.processing_times[m_id]

        self.job_ops_count = torch.tensor(job_ops_count_np, dtype=torch.long, device=self.device)
        self.op_machines_orig = torch.tensor(op_machines_np, dtype=torch.bool, device=self.device)
        self.op_proc_times_orig = torch.tensor(op_proc_times_np, dtype=torch.long, device=self.device)

        _om_f = self.op_machines_orig.float()
        _n_avail = _om_f.sum(dim=3).clamp(min=1)
        self._avg_proc_time = (self.op_proc_times_orig.float() * _om_f).sum(dim=3) / _n_avail

        masked_proc_times = torch.where(self.op_machines_orig, self.op_proc_times_orig, torch.zeros_like(self.op_proc_times_orig))
        op_max_proc_time = torch.max(masked_proc_times, dim=3).values
        self.total_operations = torch.sum(self.job_ops_count, dim=1)

        self.proc_time_scale = masked_proc_times.view(batch_size, -1).max(dim=1).values.float().clamp(min=1)

        op_mean_proc_time = masked_proc_times.sum() / masked_proc_times.gt(0).sum().clamp(min=1)
        self.max_time_estimate = (op_mean_proc_time * self.total_operations.float() / n_machines).clamp(min=1.0).clamp(min=1.0)

        self._build_graph_structure()
        self._compute_min_proc_time()

    def _build_graph_structure(self):
        """Build the precedence and succession adjacency matrices of the operations."""
        batch_size = self.batch_size
        n_jobs = self.n_jobs
        max_ops = self.max_ops_per_job
        n_ops = n_jobs * max_ops
        device = self.device

        job_ids = torch.arange(n_jobs, device=device).repeat_interleave(max_ops)
        self.opes_appertain = job_ids.unsqueeze(0).expand(batch_size, -1)
        ope_indices = torch.arange(n_ops, device=device)
        ope_in_job = ope_indices % max_ops
        count_per_op = self.job_ops_count[:, job_ids]
        self.ope_valid_mask = ope_in_job.unsqueeze(0) < count_per_op
        src_indices = torch.where(ope_in_job > 0)[0]
        dst_indices = src_indices - 1
        ope_pre_adj = torch.zeros(batch_size, n_ops, n_ops, dtype=torch.bool, device=device)
        ope_pre_adj[:, src_indices, dst_indices] = True
        valid = self.ope_valid_mask
        ope_pre_adj = ope_pre_adj & valid.unsqueeze(2) & valid.unsqueeze(1)
        self.ope_pre_adj = ope_pre_adj
        src_indices = torch.where(ope_in_job < max_ops - 1)[0]
        dst_indices = src_indices + 1
        ope_sub_adj = torch.zeros(batch_size, n_ops, n_ops, dtype=torch.bool, device=device)
        ope_sub_adj[:, src_indices, dst_indices] = True
        ope_sub_adj = ope_sub_adj & valid.unsqueeze(2) & valid.unsqueeze(1)
        self.ope_sub_adj = ope_sub_adj
        self.nums_opes = self.total_operations

    def _compute_min_proc_time(self):
        """Minimum processing time and flexibility of every operation (computed once per instance)."""
        inf_val = torch.tensor(float('inf'), device=self.device)
        masked = torch.where(
            self.op_machines_orig,
            self.op_proc_times_orig.float(),
            inf_val,
        )
        min_pt = masked.min(dim=3).values
        self.min_proc_time = torch.where(
            torch.isinf(min_pt), torch.zeros_like(min_pt), min_pt
        )
        self.op_flexibility = (
            self.op_machines_orig.sum(dim=3).float() / self.n_machines
        )

    def _compute_est_lst_slack(self):
        """Earliest start, latest start and slack of the unscheduled operations."""
        batch_size = self.batch_size
        n_jobs = self.n_jobs
        max_ops = self.max_ops_per_job
        device = self.device

        op_idx = torch.arange(max_ops, device=device)
        valid_ops = op_idx < self.job_ops_count.unsqueeze(2)
        is_remaining = (op_idx >= self.job_current_op.unsqueeze(2)) & valid_ops
        remaining_pt = self.min_proc_time * is_remaining.float()

        cumsum = remaining_pt.cumsum(dim=2)
        exclusive_cumsum = cumsum - remaining_pt
        est = self.job_available_time.unsqueeze(2) + exclusive_cumsum
        est = est * is_remaining.float()

        last_op_idx = (self.job_ops_count - 1).clamp(min=0)
        last_est = est.gather(2, last_op_idx.unsqueeze(2)).squeeze(2)
        last_min_pt = self.min_proc_time.gather(2, last_op_idx.unsqueeze(2)).squeeze(2)
        H = torch.maximum(
            (last_est + last_min_pt).max(dim=1).values,
            self.machine_available_time.max(dim=1).values,
        )

        rev_cumsum = remaining_pt.flip(2).cumsum(dim=2).flip(2)
        lst = H.unsqueeze(1).unsqueeze(2) - rev_cumsum
        lst = lst * is_remaining.float()

        slack = (lst - est).clamp(min=0) * is_remaining.float()

        return est.reshape(batch_size, -1), lst.reshape(batch_size, -1), slack.reshape(batch_size, -1)

    def _init_features(self):
        """Initial raw operation and machine features."""
        batch_size = self.batch_size
        n_jobs = self.n_jobs
        n_machines = self.n_machines
        max_ops = self.max_ops_per_job
        n_ops = n_jobs * max_ops
        device = self.device

        op_machines_flat = self.op_machines_orig.view(batch_size, n_ops, n_machines)
        op_proc_times_flat = self.op_proc_times_orig.view(batch_size, n_ops, n_machines).float()
        self.ope_ma_adj = op_machines_flat.clone()
        self.proc_time = op_proc_times_flat.clone()

        _compat = op_machines_flat.float()
        _n_compat_all = _compat.sum(dim=(1, 2)).clamp(min=1)
        self.sigma_proc = (
            (op_proc_times_flat * _compat).sum(dim=(1, 2)) / _n_compat_all
        ).clamp(min=1e-6)

        # Operation features (Section 4.4.1): 0 scheduled flag, 1 number of eligible machines, 2 mean processing time
        # (actual time once scheduled), 3 start time, 4 earliest start, 5 latest start, 6 slack
        feat_opes = torch.zeros(batch_size, 7, n_ops, dtype=torch.float32, device=device)
        feat_opes[:, 0, :] = 0
        feat_opes[:, 1, :] = op_machines_flat.sum(dim=2).float()
        n_compat = op_machines_flat.sum(dim=2).float().clamp(min=1)
        mean_proc = (op_proc_times_flat * op_machines_flat.float()).sum(dim=2) / n_compat
        feat_opes[:, 2, :] = mean_proc
        feat_opes[:, 3, :] = 0
        est_flat, lst_flat, slack_flat = self._compute_est_lst_slack()
        feat_opes[:, 4, :] = est_flat
        feat_opes[:, 5, :] = lst_flat
        feat_opes[:, 6, :] = slack_flat
        self.feat_opes = feat_opes

        # Machine features: 0 number of connected operations, 1 accumulated busy time, 2 bottleneck pressure
        # sum(1 / flexibility), 3 utilization, 4 available time
        feat_mas = torch.zeros(batch_size, 5, n_machines, dtype=torch.float32, device=device)
        feat_mas[:, 0, :] = op_machines_flat.sum(dim=1).float()
        feat_mas[:, 1, :] = 0
        inv_flex = (1.0 / self.op_flexibility.clamp(min=0.01)).view(batch_size, n_ops, 1)
        feat_mas[:, 2, :] = (op_machines_flat.float() * inv_flex).sum(dim=1)
        feat_mas[:, 3, :] = 0
        feat_mas[:, 4, :] = 0
        self.feat_mas = feat_mas

    def _init_state(self):
        """Empty partial schedule: every job and machine is available at time 0."""
        batch_size = self.batch_size
        device = self.device

        self.job_current_op = torch.zeros((batch_size, self.n_jobs), dtype=torch.long, device=device)
        self.machine_available_time = torch.zeros((batch_size, self.n_machines), dtype=torch.float32, device=device)
        self.machine_utiliz_time = torch.zeros((batch_size, self.n_machines), dtype=torch.float32, device=device)
        self.job_available_time = torch.zeros((batch_size, self.n_jobs), dtype=torch.float32, device=device)
        self.job_cumulative_proc_time = torch.zeros((batch_size, self.n_jobs), dtype=torch.float32, device=device)
        self.current_time = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.scheduled_operations = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.episode_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.makespan = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self._update_current_time()

    def _update_features(self, job_ids: torch.Tensor, current_ops: torch.Tensor, machine_ids: torch.Tensor, proc_times: torch.Tensor, start_times: torch.Tensor):
        """Update the raw features after an operation has been scheduled."""
        batch_indices = torch.arange(self.batch_size, device=self.device)
        n_jobs = self.n_jobs
        max_ops = self.max_ops_per_job
        active = ~self.episode_done

        ope_indices = job_ids * max_ops + current_ops

        batch_ope_pairs = torch.stack([batch_indices, ope_indices], dim=0)
        self.ope_ma_adj[batch_ope_pairs[0], batch_ope_pairs[1], :] = False
        self.proc_time[batch_ope_pairs[0], batch_ope_pairs[1], :] = 0

        self.ope_ma_adj[batch_indices, ope_indices, machine_ids] = active
        self.proc_time[batch_indices, ope_indices, machine_ids] = torch.where(
            active, proc_times, torch.zeros_like(proc_times)
        )

        new_status = torch.ones_like(ope_indices, dtype=torch.float32)
        self.feat_opes[batch_indices, 0, ope_indices] = torch.where(
            active, new_status, self.feat_opes[batch_indices, 0, ope_indices]
        )
        self.feat_opes[batch_indices, 2, ope_indices] = torch.where(
            active, proc_times, self.feat_opes[batch_indices, 2, ope_indices]
        )
        self.feat_opes[batch_indices, 3, ope_indices] = torch.where(
            active, start_times, self.feat_opes[batch_indices, 3, ope_indices]
        )
        est_flat, lst_flat, slack_flat = self._compute_est_lst_slack()
        self.feat_opes[:, 4, :] = est_flat
        self.feat_opes[:, 5, :] = lst_flat
        self.feat_opes[:, 6, :] = slack_flat

        self.feat_mas[:, 0, :] = self.ope_ma_adj.sum(dim=1).float()

        self.machine_utiliz_time[batch_indices, machine_ids] += torch.where(
            active, proc_times, torch.zeros_like(proc_times)
        )
        self.feat_mas[:, 1, :] = self.machine_utiliz_time

        n_ops = self.n_jobs * self.max_ops_per_job
        unscheduled = (self.feat_opes[:, 0, :] == 0).float()
        inv_flex = (1.0 / self.op_flexibility.clamp(min=0.01)).view(self.batch_size, n_ops, 1)
        self.feat_mas[:, 2, :] = (self.ope_ma_adj.float() * inv_flex * unscheduled.unsqueeze(2)).sum(dim=1)

        self.feat_mas[:, 3, :] = self.machine_utiliz_time / self.current_time.unsqueeze(1).clamp(min=1)

        self.feat_mas[:, 4, :] = self.machine_available_time

    def _compute_estimated_remaining_time(self) -> torch.Tensor:
        """Remaining mean processing time of every job."""
        batch_size = self.batch_size
        n_jobs = self.n_jobs
        max_ops = self.max_ops_per_job
        device = self.device

        avg_proc_time = self._avg_proc_time
        op_indices = torch.arange(max_ops, device=device).unsqueeze(0).unsqueeze(0)
        current_op = self.job_current_op.unsqueeze(2)
        remaining_mask = (op_indices >= current_op).float()
        valid_mask = (op_indices < self.job_ops_count.unsqueeze(2)).float()
        final_mask = remaining_mask * valid_mask
        estimated_time = (avg_proc_time * final_mask).sum(dim=2)

        return estimated_time

    def _update_current_time(self):
        """t_min: earliest start time over all candidate pairs."""
        start, _ = self._pair_start_times()
        finite = torch.isfinite(start).any(dim=(1, 2))
        new_time = start.flatten(1).min(dim=1).values
        self.current_time = torch.where(finite, new_time, self.current_time)

    def step(self, action_dict: Dict[str, torch.Tensor], start_time: torch.Tensor = None):
        """Schedule the selected pair at the start time chosen by the timing head."""
        return self._step_pair_agent(action_dict[self.pair_agent_id], start_time)

    def _step_pair_agent(self, actions: torch.Tensor, start_time: torch.Tensor = None):
        """actions: [batch] pair index (0 ~ n_jobs*n_machines-1)"""
        job_ids = actions // self.n_machines
        machine_ids = actions % self.n_machines
        safe_job_ids = torch.where(self.episode_done, torch.zeros_like(job_ids), job_ids)
        safe_machine_ids = torch.where(self.episode_done, torch.zeros_like(machine_ids), machine_ids)

        was_done = self.episode_done.clone()
        self._execute_scheduling(safe_job_ids, safe_machine_ids, start_time)

        newly_done = self.scheduled_operations >= self.total_operations
        self.episode_done = self.episode_done | newly_done
        # Terminal reward -C_max, Eq. (8) of the paper
        reward = torch.where(self.episode_done & ~was_done, -self.makespan, torch.zeros_like(self.makespan))

        obs_batch = self._get_pair_observations()
        false_tensor = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        return (
            {self.pair_agent_id: obs_batch},
            {self.pair_agent_id: reward},
            {
                self.pair_agent_id: self.episode_done.clone(),
                "__all__": self.episode_done.clone(),
            },
            {
                self.pair_agent_id: false_tensor,
                "__all__": false_tensor,
            },
            {self.pair_agent_id: {}},
        )


    def _compute_estimated_makespan(self) -> torch.Tensor:
        """Estimated makespan: maximum over the jobs of the available time plus the remaining mean processing time."""
        estimated_remaining = self._compute_estimated_remaining_time()
        job_estimated_done = self.job_available_time + estimated_remaining
        return torch.max(job_estimated_done, dim=1).values

    def _execute_scheduling(self, job_ids: torch.Tensor, machine_ids: torch.Tensor,
                            start_time: torch.Tensor = None):
        """Append the next operation of each selected job to the end of the selected machine.

        The operation starts at max(t, ES), where ES = max(T^J, T^M); for a pair from the pairing action space ES = t.
        Without a start time (non-delay variant W1S) it starts at ES. Instances that are already finished are left unchanged.
        """
        batch_indices = torch.arange(self.batch_size, device=self.device)
        active = ~self.episode_done

        safe_job_ids = torch.where(active, job_ids, torch.zeros_like(job_ids))
        safe_current_ops = self.job_current_op[batch_indices, safe_job_ids]
        safe_current_ops_clamped = torch.clamp(safe_current_ops, max=self.max_ops_per_job - 1)
        safe_machine_ids = torch.where(active, machine_ids, torch.zeros_like(machine_ids))

        processing_times = self.op_proc_times_orig[
            batch_indices, safe_job_ids, safe_current_ops_clamped, safe_machine_ids
        ].float()
        processing_times = torch.where(
            active, processing_times, torch.zeros_like(processing_times)
        )

        pair_start = torch.maximum(
            self.job_available_time[batch_indices, safe_job_ids],
            self.machine_available_time[batch_indices, safe_machine_ids],
        )
        if start_time is None:
            start_times = pair_start
        else:
            start_times = torch.maximum(start_time.to(self.device).float(), pair_start)
        start_times = torch.where(active, start_times, torch.zeros_like(start_times))
        end_times = start_times + processing_times

        self.machine_available_time[batch_indices, safe_machine_ids] = torch.where(
            active, end_times, self.machine_available_time[batch_indices, safe_machine_ids])
        self.job_available_time[batch_indices, safe_job_ids] = torch.where(
            active, end_times, self.job_available_time[batch_indices, safe_job_ids])
        self.job_cumulative_proc_time[batch_indices, safe_job_ids] += torch.where(
            active, processing_times, torch.zeros_like(processing_times)
        )
        self.job_current_op[batch_indices, safe_job_ids] = torch.where(
            active, self.job_current_op[batch_indices, safe_job_ids] + 1,
            self.job_current_op[batch_indices, safe_job_ids])
        self.scheduled_operations = torch.where(
            active, self.scheduled_operations + 1, self.scheduled_operations
        )

        self._pair_start_cache = None
        self._update_current_time()
        self._update_features(safe_job_ids, safe_current_ops, safe_machine_ids, processing_times, start_times)

        self.makespan = torch.max(self.machine_available_time, dim=1).values

    def _get_pair_observations(self) -> Dict[str, torch.Tensor]:
        """Observation: mask of all candidate pairs [batch, n_jobs * n_machines]."""
        return {"action_mask": self._pair_mask_all()}

    def _pair_mask_all(self) -> torch.Tensor:
        """Mask of all candidate pairs, regardless of their start time."""
        _, valid = self._pair_start_times()
        return valid.reshape(self.batch_size, -1).float()

    def _compute_pair_action_mask(self, query_time: torch.Tensor) -> torch.Tensor:
        """Pairing action space: candidate pairs whose earliest start time equals t."""
        start, valid = self._pair_start_times()
        t = query_time.to(self.device).float().view(-1, 1, 1)
        hit = valid & (start == t)
        return hit.reshape(self.batch_size, -1).float()

    def can_schedule(self) -> torch.Tensor:
        """Whether a candidate pair is left [batch]."""
        return (self._pair_mask_all().sum(dim=1) > 0) & (~self.episode_done)

    def _pair_start_times(self):
        """Earliest start time max(job available time, machine available time) and validity of every (job, machine) pair."""
        if self._pair_start_cache is not None:
            return self._pair_start_cache
        job_completed = self.job_current_op >= self.job_ops_count
        cur_ops = torch.clamp(self.job_current_op, max=self.max_ops_per_job - 1)
        cur_idx = cur_ops.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.n_machines)
        can_process = torch.gather(self.op_machines_orig, 2, cur_idx).squeeze(2)

        valid = can_process & (~job_completed).unsqueeze(2) & (~self.episode_done).view(-1, 1, 1)
        start = torch.maximum(
            self.job_available_time.unsqueeze(2),
            self.machine_available_time.unsqueeze(1),
        )
        inf = torch.tensor(float('inf'), device=self.device)
        result = (torch.where(valid, start, inf), valid)
        self._pair_start_cache = result
        return result

    def get_global_features(self) -> torch.Tensor:
        """Global features of the critic [batch, 2]: progress and estimated makespan divided by the mean processing time."""
        progress = self.scheduled_operations.float() / self.total_operations.float().clamp(min=1)
        lb = self._compute_estimated_makespan() / self.sigma_proc
        return torch.stack([progress, lb], dim=-1)

    def get_candidate_scheduling_times(self):
        """Timing action space: distinct earliest start times of the candidate pairs that are earlier than C*."""
        bs = self.batch_size
        device = self.device
        n_ops = self.n_jobs * self.max_ops_per_job
        n_jobs, n_mas = self.n_jobs, self.n_machines

        start, valid = self._pair_start_times()

        # Every candidate start time equals a job or a machine available time.
        times = torch.cat([self.job_available_time, self.machine_available_time], dim=1)
        max_cand = times.shape[1]

        hit = valid.unsqueeze(1) & (start.unsqueeze(1) == times.view(bs, max_cand, 1, 1))
        cand_mask = hit.any(dim=-1).any(dim=-1) & (~self.episode_done).unsqueeze(1)

        dup = (times.unsqueeze(2) == times.unsqueeze(1)).triu(diagonal=1).any(dim=1)
        cand_mask = cand_mask & (~dup)

        # Jobs / machines that have a candidate pair starting at each candidate time
        ready_job = hit.any(dim=-1)
        idle_ma_mask = hit.any(dim=2)
        op_indices = torch.arange(n_jobs, device=device) * self.max_ops_per_job \
            + torch.clamp(self.job_current_op, max=self.max_ops_per_job - 1)
        ready_op_mask = torch.zeros(bs, max_cand, n_ops, dtype=torch.bool, device=device)
        ready_op_mask.scatter_(2, op_indices.unsqueeze(1).expand(-1, max_cand, -1), ready_job)

        INF = torch.finfo(times.dtype).max
        p_cur = self.proc_time.gather(
            1, op_indices.unsqueeze(-1).expand(-1, -1, n_mas))
        completion = torch.where(valid, start + p_cur,
                                 torch.full_like(start, INF))
        # Earliest-completion bound C*: keep only start times earlier than the earliest completion time.
        c_star = completion.view(bs, -1).min(dim=1).values
        cand_mask = cand_mask & (times < c_star.unsqueeze(1))

        sf = self._candidate_scalar_feats(times, cand_mask, hit, ready_job, idle_ma_mask)
        return {
            'times': times,
            'mask': cand_mask,
            'scalar_feats': sf,
            'ready_op_mask': ready_op_mask,
            'idle_ma_mask': idle_ma_mask,
        }

    def _candidate_scalar_feats(self, times, cand_mask, hit, ready_job, idle_ma_mask):
        """Scalar features of the candidate start times [batch, C, 9]; times are divided by the mean processing time."""
        bs, max_cand = times.shape
        device = self.device
        sigma = self.sigma_proc.clamp(min=1e-6).view(bs, 1)

        start, valid = self._pair_start_times()
        cur_ops = torch.clamp(self.job_current_op, max=self.max_ops_per_job - 1)
        cur_idx = cur_ops.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.n_machines)
        proc = torch.gather(self.op_proc_times_orig, 2, cur_idx).squeeze(2).float()

        inf = torch.tensor(float('inf'), device=device)
        t_min = torch.where(cand_mask, times, inf).min(dim=1).values.view(bs, 1)
        t_min = torch.where(torch.isinf(t_min), torch.zeros_like(t_min), t_min)

        n_valid = valid.sum(dim=(1, 2)).clamp(min=1).view(bs, 1).float()
        sf = torch.zeros(bs, max_cand, 9, device=device)

        # Features 1-9 of Table 3 in the paper
        sf[:, :, 0] = (times - t_min) / sigma                                # waiting time
        sf[:, :, 1] = hit.sum(dim=(2, 3)).float() / n_valid                   # pair ratio
        sf[:, :, 2] = ready_job.sum(dim=-1).float() / self.n_jobs             # job ratio
        sf[:, :, 3] = idle_ma_mask.sum(dim=-1).float() / self.n_machines      # machine ratio

        inf_p = torch.tensor(float('inf'), device=device)
        proc_at_t = torch.where(hit, proc.unsqueeze(1), inf_p)
        best_at_t = proc_at_t.amin(dim=(2, 3))
        has_any = torch.isfinite(best_at_t)
        best_at_t = torch.where(has_any, best_at_t, torch.zeros_like(best_at_t))
        at_min = torch.where(cand_mask & (times == t_min), best_at_t, inf_p).min(dim=1).values
        at_min = torch.where(torch.isinf(at_min), torch.zeros_like(at_min), at_min).view(bs, 1)
        best_global = torch.where(valid, proc, inf_p).amin(dim=(1, 2))
        best_global = torch.where(torch.isinf(best_global),
                                  torch.zeros_like(best_global), best_global).view(bs, 1)

        sf[:, :, 4] = best_at_t / sigma                                       # best processing time
        cnt = hit.sum(dim=(2, 3)).clamp(min=1).float()
        sf[:, :, 5] = torch.where(hit, proc.unsqueeze(1), torch.zeros_like(proc).unsqueeze(1)
                                  ).sum(dim=(2, 3)) / cnt / sigma             # mean processing time
        sf[:, :, 6] = (best_at_t <= best_global + 1e-6).float() * has_any.float()  # global-best indicator
        sf[:, :, 7] = (at_min - best_at_t).clamp(min=0) / sigma              # processing-time gain over t_min
        sf[:, :, 8] = sf[:, :, 7] - sf[:, :, 0]                               # net gain

        return sf * cand_mask.unsqueeze(-1).float()

    def get_job_op_indices(self) -> torch.Tensor:
        """Index of the next operation of every job [batch, n_jobs] (last operation for finished jobs)."""
        batch_indices = torch.arange(self.batch_size, device=self.device).unsqueeze(1)
        job_indices = torch.arange(self.n_jobs, device=self.device).unsqueeze(0)

        current_ops = torch.clamp(self.job_current_op, max=self.max_ops_per_job - 1)
        job_op_indices = job_indices * self.max_ops_per_job + current_ops

        return job_op_indices

    def get_network_input(self) -> Dict[str, torch.Tensor]:
        """Graph input of the encoder (node features, processing times on the edges, adjacency, masks)."""
        return {
            "ope_feats": self.feat_opes.transpose(1, 2).clone(),
            "ma_feats": self.feat_mas.transpose(1, 2).clone(),
            "proc_time": self.proc_time.clone(),
            "ope_ma_adj": self.ope_ma_adj.clone(),
            "ope_pre_adj": self.ope_pre_adj.clone(),
            "ope_sub_adj": self.ope_sub_adj.clone(),
            "nums_opes": self.nums_opes,
            "ope_valid_mask": self.ope_valid_mask.clone(),
        }

    def get_pair_private_features(self, query_time: torch.Tensor) -> torch.Tensor:
        """Features of every (job, machine) pair at time t [batch, n_jobs * n_machines, 9] (Table 4 of the paper).

        0 remaining operations, 1 remaining mean work, 2 accumulated waiting, 3 completion ratio,
        4 minimum processing time of the next operation, 5 number of its eligible machines,
        6 processing time of the current operation on the machine, 7 estimated start max(T^J, T^M, t),
        8 estimated completion.
        """
        qt = query_time.to(self.device).float()
        bs = self.batch_size
        n_jobs = self.n_jobs
        n_machines = self.n_machines
        n_pairs = n_jobs * n_machines
        device = self.device

        feats = torch.zeros(bs, n_jobs, n_machines, 9, device=device, dtype=torch.float32)

        current_ops_clamped = torch.clamp(self.job_current_op, max=self.max_ops_per_job - 1)

        remaining_ops = (self.job_ops_count - self.job_current_op).float()
        feats[:, :, :, 0] = remaining_ops.unsqueeze(2)

        estimated_remaining = self._compute_estimated_remaining_time()
        feats[:, :, :, 1] = estimated_remaining.unsqueeze(2)

        wait_time = self.job_available_time - self.job_cumulative_proc_time
        feats[:, :, :, 2] = wait_time.unsqueeze(2)

        completion_ratio = self.job_current_op.float() / self.job_ops_count.float().clamp(min=1)
        feats[:, :, :, 3] = completion_ratio.unsqueeze(2)

        next_op_idx = torch.clamp(self.job_current_op + 1, max=self.max_ops_per_job - 1)
        has_next = (self.job_current_op + 1) < self.job_ops_count
        next_idx_exp = next_op_idx.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, n_machines)
        next_op_machines = torch.gather(self.op_machines_orig, 2, next_idx_exp).squeeze(2)
        next_op_proc = torch.gather(self.op_proc_times_orig, 2, next_idx_exp).squeeze(2).float()
        inf_val = torch.tensor(float('inf'), device=device)
        next_min = torch.where(next_op_machines, next_op_proc, inf_val).min(dim=2).values
        next_min = torch.where(torch.isinf(next_min), torch.zeros_like(next_min), next_min)
        next_min = next_min * has_next.float()
        feats[:, :, :, 4] = next_min.unsqueeze(2)

        next_n_machines = (next_op_machines.sum(dim=2).float() * has_next.float())
        feats[:, :, :, 5] = next_n_machines.unsqueeze(2)

        cur_idx_exp = current_ops_clamped.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, n_machines)
        cur_op_proc = torch.gather(self.op_proc_times_orig, 2, cur_idx_exp).squeeze(2).float()

        feats[:, :, :, 6] = cur_op_proc

        start_time = torch.maximum(
            torch.maximum(
                self.job_available_time.unsqueeze(2),
                self.machine_available_time.unsqueeze(1),
            ),
            qt.unsqueeze(1).unsqueeze(2),
        )
        feats[:, :, :, 7] = start_time

        feats[:, :, :, 8] = start_time + cur_op_proc

        return feats.view(bs, n_pairs, 9)

    def get_makespans(self) -> torch.Tensor:
        """Makespan of the current (partial) schedules [batch]."""
        return torch.max(self.machine_available_time, dim=1).values
