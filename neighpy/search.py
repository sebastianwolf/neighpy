import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple, Protocol, Union
from joblib import Parallel, delayed
from tqdm import tqdm
from os import cpu_count
import os
import shutil

import time
import subprocess
import json
import re


def jobs_running_slurm(job_ids):
    result = subprocess.run(["squeue", "--noheader", "--jobs", ",".join(job_ids)],
                            capture_output=True, text=True)
    return bool(result.stdout.strip())


def jobs_running(job_ids,scheduler):

    if scheduler == "pbs":
        result = subprocess.run(["qstat"], capture_output=True, text=True)
        # qstat output contains job IDs in the first column
        # be careful here, this very much depends on the server that you are running on.
        running_ids = {line.split()[0] for line in result.stdout.strip().splitlines()[2:]}  # skip header
        running_job_numbers = {rid.split('.', 1)[0] for rid in running_ids}
        return any(jid in running_job_numbers for jid in job_ids)
    elif scheduler == "slurm":
        result = subprocess.run(["squeue", "--noheader", "--jobs", ",".join(job_ids)],
            capture_output=True, text=True)
        return bool(result.stdout.strip())
        # result = subprocess.run(["squeue", "--noheader", "-o", "%A"], capture_output=True, text=True)
        # running_ids = set(result.stdout.strip().splitlines())
        # return any(jid in running_ids for jid in job_ids)
    else:
        raise RuntimeError("No valid scheduler provided.")


def read_misfit(filename="misfit.txt"):
    """Read misfit value from file and return as float."""
    with open(filename, "r") as f:
        return float(f.read().strip())

class ObjectiveFunction(Protocol):
    """
    :meta private:
    """

    def __call__(self, x: NDArray, *args: Any) -> float:
        # Any type hint because the objective function supplied by the user can have any signature
        # as long as its first argument is of type NDArray and returns a float
        # A type hint of Callable[[NDArray], float] would be too restrictive
        # A type checker will probably complain about this, but it's the best we can do for now
        # From python 3.10, we can use ParamSpec to define a generic type hint for the objective
        # function, and remove this protocol
        #
        # Example:
        # P = ParamSpec("P")
        # class NASearcher:
        #     def __init__(self, objective: Callable[Concatenate[NDArray, P], float]):
        ...


class NASearcher:
    """
    Args:
        objective (Callable[[NDArray], float]): The objective function to minimize.
            This function should take a single argument of type NDArray and return a float.
        ns (int): The number of samples generated at each iteration.
        nr (int): The number of cells to resample.
        ni (int): The number of samples from initial random search.
        n (int): The number of iterations.
        bounds (Tuple[Tuple[float, float], ...]): A tuple of tuples representing the bounds of the search space.
            Each inner tuple represents the lower and upper bounds for a specific dimension.
        args (Tuple, optional): Additional arguments to pass to the objective function.
        seed (int, optional): Seed for the random number generator.
    """

    def __init__(
        self,
        objective: ObjectiveFunction,
        ns: int,
        nr: int,
        ni: int,
        n: int,
        bounds: Tuple[Tuple[float, float], ...],
        args: Tuple = (),
        seed: Union[int, None] = None,
    ) -> None:
        self._objective = objective
        self.objective_args = args
        self.scheduler = None

        self.ns = ns  # number of samples generated at each iteration
        self.nr = nr  # number of cells to resample
        self.nspnr = ns // nr  # number of samples per cell to generate
        self.ni = ni  # number of samples from initial random search
        self.n = n  # number of iterations
        self.nt = ni + n * ns  # total number of samples
        self.np = 0  # running total of number of samples

        self.parentdir = os.getcwd()
        self.singlerundir = self.parentdir + '/FwdModels'
        self.bounds = bounds  # bounds of the search space
        self.nd = len(bounds)  # number of dimensions
        self.lower = np.array([b[0] for b in bounds])
        self.upper = np.array([b[1] for b in bounds])
        self.Cm = (
            1 / (self.upper - self.lower) ** 2
        )  # (diagonal) prior covariance matrix

        self.samples = np.zeros((self.nt, self.nd))
        self.objectives = np.full(
            self.nt, np.inf
        )  # start with inf since we want to minimize
        self._current_best_ind = 0

        ss = np.random.SeedSequence(seed)
        self.rngs = [np.random.default_rng(s) for s in ss.spawn(self.nr)]

    def run(self, parallel=True) -> None:
        """
        Run the Direct Search.

        Populates the following attributes:

        - **samples** (`NDArray`) - samples generated by the direct search.
        - **objectives** (`NDArray`) - objective function values for each sample.

        """
        # initial random search
        print("NAI - Initial Random Search")
        new_samples = self._initial_random_search()
        self._update_ensemble(new_samples)

        n_jobs = min(self.nr, cpu_count()) if parallel else 1
        print("cpu_count is",cpu_count())
        with Parallel(n_jobs=n_jobs) as _parallel:
            # main optimisation loop
            for _ in tqdm(range(self.n), desc="NAI - Optimisation Loop"):
                inds = self._get_best_indices()
                self._current_best_ind = inds[0]
                cells_to_resample = self.samples[inds]

                t1 = time.time()
                new_samples = _parallel(
                    delayed(self._random_walk_in_voronoi)(cell, k, rng)
                    for k, cell, rng in zip(inds, cells_to_resample, self.rngs)
                )
                t2 = time.time()
                
                self._update_ensemble(np.concatenate(new_samples))
                t3 = time.time()

                print(f"Execution time random walk: {t2 - t1:.6f} seconds")
                print(f"Execution time update ensamble: {t3 - t2:.6f} seconds")

    def run_external(self, parallel=True, path='.') -> None:
        """
        Run the Direct Search.

        Populates the following attributes:

        - **samples** (`NDArray`) - samples generated by the direct search.
        - **objectives** (`NDArray`) - objective function values for each sample.

        """
        # initial random search
        print("NAI - Initial Random Search")
        new_samples = self._initial_random_search()
        self._update_ensemble_external(path,new_samples)

        n_jobs = min(self.nr, cpu_count()) if parallel else 1
        print("cpu_count is",cpu_count())
        with Parallel(n_jobs=n_jobs) as _parallel: # this loop only parallelises the voronoi parameter search. Not needed in case the objective function takes MOST of the computational time.
            # main optimisation loop
            for _ in range(self.n):
                inds = self._get_best_indices()
                self._current_best_ind = inds[0]
                cells_to_resample = self.samples[inds]

                t1 = time.time()
                new_samples = _parallel(
                    delayed(self._random_walk_in_voronoi)(cell, k, rng)
                    for k, cell, rng in zip(inds, cells_to_resample, self.rngs)
                )
                t2 = time.time()
                
                self._update_ensemble_external(path,np.concatenate(new_samples))
                t3 = time.time()
                # print("new_samples",new_samples)
                # print("np.concatenate(new_samples)",np.concatenate(new_samples))
                print(f"Execution time random walk: {t2 - t1:.6f} seconds")
                print(f"Execution time update ensemble: {t3 - t2:.6f} seconds")

    def run_external_queue(self, parallel=True, restart=False) -> None:
        """
        Run the Direct Search.

        Populates the following attributes:

        - **samples** (`NDArray`) - samples generated by the direct search.
        - **objectives** (`NDArray`) - objective function values for each sample.

        """
        # check which scheduler is used:
        if shutil.which("sbatch"):
            self.scheduler = "slurm"
            print("Detected slurm scheduler for job submission. Using slurm submission.")
        elif shutil.which("qsub"):
            self.scheduler = "pbs"
            print("Detected PBS scheduler for job submission. Using slurm submission.")
        else:
            raise RuntimeError("Neither Slurm (sbatch) nor PBS (qsub) detected on this system.")

        # initial random search
        if not restart:
            print("NAI - Initial Random Search")
            new_samples = self._initial_random_search()
            self._update_ensemble_external_queue(new_samples)
        else:
            print("Restarting NAI - skipping initial random search and using loaded samples and objectives as initial population.")

        n_jobs = min(self.nr, cpu_count()) if parallel else 1
        # print("cpu_count is",cpu_count())
        with Parallel(n_jobs=n_jobs) as _parallel: # this loop only parallelises the voronoi parameter search. Not needed in case the objective function takes MOST of the computational time.
            # main optimisation loop
            for _ in range(self.n):
                inds = self._get_best_indices()
                self._current_best_ind = inds[0]
                cells_to_resample = self.samples[inds]

                t1 = time.time()
                new_samples = _parallel(
                    delayed(self._random_walk_in_voronoi)(cell, k, rng)
                    for k, cell, rng in zip(inds, cells_to_resample, self.rngs)
                )
                t2 = time.time()
                
                self._update_ensemble_external_queue(np.concatenate(new_samples))
                t3 = time.time()
                # print("new_samples",new_samples)
                # print("np.concatenate(new_samples)",np.concatenate(new_samples))
                print(f"Execution time random walk: {t2 - t1:.6f} seconds")
                print(f"Execution time update ensemble: {t3 - t2:.6f} seconds")



    def objective(self, x: NDArray) -> float:
        return self._objective(x, *self.objective_args)

    def _initial_random_search(self) -> NDArray:
        return self.rngs[0].uniform(
            low=self.lower,
            high=self.upper,
            size=(self.ni, self.nd),
        )

    def _random_walk_in_voronoi(
        self, vk: NDArray, k: int, rng: np.random.Generator
    ) -> NDArray:
        # FOLLOWING https://github.com/underworldcode/pyNA/blob/30d1cb7955d6b1389eae885127389ed993fa6940/pyNA/sampler.py#L85

        # vk is the current voronoi cell
        # k is the index of the current voronoi cell

        new_samples = np.empty((self.nspnr, self.nd))
        old_samples = self.samples[: self.np]
        walk_length = self.nspnr
        if k == self._current_best_ind:
            # best model so walk a bit further
            walk_length += self.ns % self.nr

        # find cell boundaries along each dimension
        d2 = np.sum(
            self.Cm * (vk - old_samples) ** 2, axis=1
        )  # distance to all other cells

        d2_previous_axis = 0  # distance to previous axis

        for _step in range(walk_length):
            xA = vk.copy()  # start of walk at cell centre
            for i in range(self.nd):  # step along each axis
                d2_current_axis = self.Cm[i] * (xA[i] - old_samples[:, i]) ** 2
                d2 += d2_previous_axis - d2_current_axis
                dk2 = d2[k]  # disctance of cell centre to axis

                # eqn (19) Sambridge 1999
                vji = old_samples[:, i]
                vki = vk[i]
                a = dk2 - d2
                b = vki - vji
                xji = 0.5 * (
                    vki + vji + np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                )

                # eqns (20, 21) Sambridge 1999
                li = np.nanmax(np.hstack((self.lower[i], xji[xji < xA[i]])))
                ui = np.nanmin(np.hstack((self.upper[i], xji[xji > xA[i]])))
                xA[i] = rng.uniform(li, ui)

                d2_previous_axis = d2_current_axis

            new_samples[_step] = xA

        return new_samples

    def _get_best_indices(self) -> NDArray:
        # there may be a faster way to do this using np.argpartition
        return np.argsort(self.objectives)[: self.nr]

    def _update_ensemble(self, new_samples: NDArray):
        n = new_samples.shape[0]
        self.samples[self.np : self.np + n] = new_samples
        self.objectives[self.np : self.np + n] = np.apply_along_axis(
            self.objective, 1, new_samples
        )
        self.np += n

    def _update_ensemble_external(self, path,new_samples: NDArray):
        n = new_samples.shape[0]
        self.samples[self.np : self.np + n] = new_samples
        # print(new_samples[0])
        procs = []
        for i in range(n):
            proc = subprocess.Popen(["python","external_rosenbrock.py",str(tuple(new_samples[i])),str(i)]) 
            procs.append(proc)
        for p in procs:
            p.wait()
        for i in range(n):  
            with open(path + "/output"+str(i)+".json", "r") as f:
                data = json.load(f)
            misfit_i = data["misfit"]
            self.objectives[self.np+i] = misfit_i
            
        self.np += n

    def _update_ensemble_external_queue(self, new_samples: NDArray):
        # Submit SLURM jobs and collect job IDs
        n = new_samples.shape[0]
        self.samples[self.np : self.np + n] = new_samples
        job_ids = []
        print('new_samples',new_samples)
        script =  self.parentdir + '/' + 'SingleRun.sh'

        #!!! submit jobs
        # first create new folders, then submit jobs
        for i in range(n):
            iterationstr = 'Model'+str(self.np + i)
            os.makedirs(self.singlerundir + '/' + iterationstr, exist_ok=True) # make new folder for model
            with open(self.singlerundir + '/' + iterationstr +'/variables.txt', 'w') as f: # write new_sample to file
                f.write(str(new_samples[i]))

        for i in range(n):
            iterationstr = 'Model'+str(self.np + i)
            print('submitting job with i=',i)
            if self.scheduler == "pbs":
                result = subprocess.run(["qsub", '-F', f"{self.parentdir} {self.singlerundir} {iterationstr}", script], capture_output=True, text=True) # this script needs to run a job, do post-proc, and copy the resulting misfit back to the parent folder
                match = re.match(r"(\d+)", result.stdout.strip())
            elif self.scheduler == "slurm":
                # Slurm returns: "Submitted batch job 123456"
                result = subprocess.run(["sbatch", "--export=ALL", script, self.parentdir, self.singlerundir, iterationstr], capture_output=True, text=True)  # this script needs to run a job, do post-proc, and copy the resulting misfit back to the parent folder
                #time.sleep(5)
                match = re.search(r"Submitted batch job (\d+)", result.stdout)
            else:
                raise RuntimeError("Neither Slurm (sbatch) nor PBS (qsub) scheduler provided.")


            if match:
                job_id = match.group(1)
                job_ids.append(job_id)
                print(f"Submitted {script} as job {job_id}")
            else:
                print(f"Failed to submit {script}:\n{result.stderr}")
                print(result.stderr)

         #!!! Wait that all jobs are completed
        print("Waiting for jobs to finish...")
        while jobs_running(job_ids,self.scheduler): # slurm or pbs
            # print("Still running...")
            time.sleep(30)  # Wait 10 seconds before checking again
        print("All queue jobs completed.")

        #!!! Load computed misfit and populate self.objectives with computed misfits
        for i in range(n):
            filename = self.singlerundir + '/Model'+str(self.np + i) + '/misfit.txt'
            loaded_misfit = read_misfit(filename=filename)
            self.objectives[self.np+i] = loaded_misfit
            
        self.np += n

        #!!! Now next iteration of parameter-search can start.
