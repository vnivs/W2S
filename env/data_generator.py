import os
import random
import numpy as np
from typing import List, Optional
from utils.instance_parser import FJSPInstance, Job, Operation


class FJSPGenerator:
    """Random FJSP instance generator."""

    def __init__(
        self,
        n_jobs: int,
        n_machines: int,
        ops_per_job_min: int = 1,
        ops_per_job_max: int = 10,
        machines_per_op_min: int = 1,
        machines_per_op_max: Optional[int] = None,
        proc_time_min: int = 1,
        proc_time_max: int = 20,
        proc_time_std: float = 0.2,
        output_dir: Optional[str] = None,
        seed: Optional[int] = None
    ):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.ops_per_job_min = ops_per_job_min
        self.ops_per_job_max = ops_per_job_max
        self.machines_per_op_min = machines_per_op_min
        self.machines_per_op_max = machines_per_op_max or n_machines
        self.proc_time_min = proc_time_min
        self.proc_time_max = proc_time_max
        self.proc_time_std = proc_time_std
        self.output_dir = output_dir

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __call__(self, idx: int = 0) -> FJSPInstance:
        """Generate one instance."""
        instance = FJSPInstance(self.n_jobs, self.n_machines)
        lines = [f"{self.n_jobs} {self.n_machines}"]

        for job_id in range(self.n_jobs):
            job = Job(job_id)
            n_ops = random.randint(self.ops_per_job_min, self.ops_per_job_max)
            line_parts = [str(n_ops)]

            for op_id in range(n_ops):
                operation = Operation(job_id, op_id)
                n_machines_available = random.randint(
                    self.machines_per_op_min,
                    min(self.machines_per_op_max, self.n_machines)
                )
                available_machines = sorted(random.sample(
                    range(self.n_machines), n_machines_available
                ))
                line_parts.append(str(n_machines_available))

                mean_time = random.randint(self.proc_time_min, self.proc_time_max)
                for machine_id in available_machines:
                    low = max(self.proc_time_min, round(mean_time * (1 - self.proc_time_std)))
                    high = min(self.proc_time_max, round(mean_time * (1 + self.proc_time_std)))
                    proc_time = random.randint(low, high)
                    operation.add_machine(machine_id, proc_time)
                    line_parts.append(str(machine_id))
                    line_parts.append(str(proc_time))

                job.add_operation(operation)

            instance.add_job(job)
            lines.append(" ".join(line_parts))

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            filename = f"{self.n_jobs}j_{self.n_machines}m_{str(idx).zfill(3)}.txt"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w') as f:
                f.write("\n".join(lines))

        return instance

    def generate_batch(self, batch_size: int, start_idx: int = 0) -> List[FJSPInstance]:
        """Generate a list of instances."""
        return [self.__call__(start_idx + i) for i in range(batch_size)]
