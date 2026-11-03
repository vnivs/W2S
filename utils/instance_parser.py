"""Parser for FJSP benchmark instances."""

from typing import List, Dict


class Operation:
    """An operation."""

    def __init__(self, job_id: int, op_id: int):
        self.job_id = job_id
        self.op_id = op_id
        self.available_machines: List[int] = []
        self.processing_times: Dict[int, int] = {}

    def add_machine(self, machine_id: int, processing_time: int):
        """Add an eligible machine with its processing time."""
        self.available_machines.append(machine_id)
        self.processing_times[machine_id] = processing_time

    def get_min_processing_time(self) -> int:
        """Minimum processing time."""
        return min(self.processing_times.values())

    def get_max_processing_time(self) -> int:
        """Maximum processing time."""
        return max(self.processing_times.values())

    def __repr__(self):
        return f"Op(J{self.job_id}O{self.op_id}, machines={self.available_machines})"


class Job:
    """A job."""

    def __init__(self, job_id: int):
        self.job_id = job_id
        self.operations: List[Operation] = []
        self.current_op_idx = 0

    def add_operation(self, operation: Operation):
        """Add an operation."""
        self.operations.append(operation)

    @property
    def num_operations(self) -> int:
        """Number of operations."""
        return len(self.operations)

    @property
    def is_completed(self) -> bool:
        """Whether all operations are completed."""
        return self.current_op_idx >= self.num_operations

    @property
    def current_operation(self) -> Operation:
        """Next operation to be processed."""
        if self.is_completed:
            return None
        return self.operations[self.current_op_idx]

    def __repr__(self):
        return f"Job{self.job_id}(ops={self.num_operations}, current={self.current_op_idx})"


class FJSPInstance:
    """An FJSP instance."""

    def __init__(self, n_jobs: int, n_machines: int):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.jobs: List[Job] = []

    def add_job(self, job: Job):
        """Add a job."""
        self.jobs.append(job)

    def get_total_operations(self) -> int:
        """Total number of operations."""
        return sum(job.num_operations for job in self.jobs)

    def get_lower_bound_makespan(self) -> int:
        """Lower bound on the makespan: sum of the minimum processing times divided by the number of machines."""
        total_min_time = sum(
            op.get_min_processing_time()
            for job in self.jobs
            for op in job.operations
        )
        return total_min_time // self.n_machines

    def __repr__(self):
        return f"FJSPInstance(jobs={self.n_jobs}, machines={self.n_machines}, total_ops={self.get_total_operations()})"


def _detect_zero_based_fjsp(lines: List[str], n_jobs: int) -> bool:
    """Check whether the machines of a standard FJSP file are numbered from 0."""
    all_machine_ids: List[int] = []
    for line in lines[1:1 + n_jobs]:
        tokens = line.strip().split()
        idx = 0
        n_operations = int(tokens[idx]); idx += 1
        for _ in range(n_operations):
            n_machines_for_op = int(tokens[idx]); idx += 1
            for _ in range(n_machines_for_op):
                all_machine_ids.append(int(tokens[idx])); idx += 1
                idx += 1
    return min(all_machine_ids) == 0 if all_machine_ids else True


def _parse_fjsp_standard(lines: List[str], n_jobs: int, n_machines: int) -> FJSPInstance:
    """Parse a file in the standard FJSP format."""
    instance = FJSPInstance(n_jobs, n_machines)

    is_zero_based = _detect_zero_based_fjsp(lines, n_jobs)

    for job_id in range(n_jobs):
        line = lines[job_id + 1].strip().split()
        idx = 0

        n_operations = int(line[idx])
        idx += 1

        job = Job(job_id)

        for op_id in range(n_operations):
            operation = Operation(job_id, op_id)

            n_machines_for_op = int(line[idx])
            idx += 1

            for _ in range(n_machines_for_op):
                machine_id = int(line[idx])
                if not is_zero_based:
                    machine_id -= 1
                idx += 1
                processing_time = int(line[idx])
                idx += 1

                operation.add_machine(machine_id, processing_time)

            job.add_operation(operation)

        instance.add_job(job)

    return instance


def _looks_like_taillard_jsp(job_lines: List[List[str]], n_jobs: int, n_machines: int) -> bool:
    """Check whether a file is in the Taillard-style JSP format."""
    if len(job_lines) < n_jobs:
        return False

    for tokens in job_lines[:n_jobs]:
        if len(tokens) != 2 * n_machines:
            return False
        try:
            values = [int(x) for x in tokens]
        except ValueError:
            return False

        machine_ids = values[0::2]
        proc_times = values[1::2]

        if any(p <= 0 for p in proc_times):
            return False

        zero_based_ok = all(0 <= m < n_machines for m in machine_ids)
        one_based_ok = all(1 <= m <= n_machines for m in machine_ids)
        if not (zero_based_ok or one_based_ok):
            return False

    return True


def _parse_taillard_jsp(lines: List[str], n_jobs: int, n_machines: int) -> FJSPInstance:
    """Parse a Taillard-style JSP file into an FJSPInstance."""
    instance = FJSPInstance(n_jobs, n_machines)
    raw_job_lines = [lines[job_id + 1].strip().split() for job_id in range(n_jobs)]

    all_machine_ids = []
    for tokens in raw_job_lines:
        values = [int(x) for x in tokens]
        all_machine_ids.extend(values[0::2])

    is_zero_based = min(all_machine_ids) == 0

    for job_id, tokens in enumerate(raw_job_lines):
        values = [int(x) for x in tokens]
        job = Job(job_id)

        for op_id in range(n_machines):
            machine_id = values[2 * op_id]
            processing_time = values[2 * op_id + 1]

            if not is_zero_based:
                machine_id -= 1

            operation = Operation(job_id, op_id)
            operation.add_machine(machine_id, processing_time)
            job.add_operation(operation)

        instance.add_job(job)

    return instance


def parse_fjsp_instance(file_path: str) -> FJSPInstance:
    """Parse an FJSP or JSP instance file."""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    first_line = lines[0].split()
    n_jobs = int(first_line[0])
    n_machines = int(first_line[1])

    job_lines = [line.split() for line in lines[1:1 + n_jobs]]

    if _looks_like_taillard_jsp(job_lines, n_jobs, n_machines):
        return _parse_taillard_jsp(lines, n_jobs, n_machines)

    return _parse_fjsp_standard(lines, n_jobs, n_machines)
