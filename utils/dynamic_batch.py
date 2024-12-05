import psutil
from tqdm import tqdm
import time
from typing import List, Optional, Union
import pandas as pd

class DynamicBatchIterator:
    """
    iterator to use with joblib parallel that allows to set
    a specific cpu load target. Also comes with a nice
    progress bar.
    """
    def __init__(self, df: pd.DataFrame, n_cpus: int, min_batch: int = 50, max_batch: int = 500, 
                 target_cpu: int = 90, crop_df: bool = False, tqdmlogger: Optional[Union[str, 'tqdm']] = None):
        self.df: pd.DataFrame = df
        self.todo: List[Union[str, int]] = self.df.index.tolist()
        self.max_batch: int = max_batch
        self.min_batch: int = min_batch
        self.target_cpu: int = target_cpu
        self.n_cpus: int = n_cpus
        self.current_batch_size: int = min_batch
        self.pbar: tqdm = tqdm(
                total=len(self.todo),
                desc="Processing users",
                smoothing=0.0,
                file=tqdmlogger,
                )
        self.latest: float = time.time()
        self.crop_df: bool = crop_df

    def __iter__(self) -> 'DynamicBatchIterator':
        return self

    def __next__(self) -> pd.DataFrame:
        if len(self.todo) == 0:
            self.pbar.close()
            raise StopIteration

        # only adjust parameters every X seconds
        if time.time() - self.latest > 30:
            self._adjust_cpu()

        # Get next batch
        batch_ids = self.todo[:self.current_batch_size]
        if self.crop_df:
            # reduce df size at each new iteration
            batch = self.df.loc[batch_ids].copy()
            self.df = self.df.drop(index=batch_ids)
        else:
            # keep the whole df and only slice it
            batch = self.df.loc[batch_ids]
        self.pbar.update(len(batch_ids))
        [self.todo.remove(t) for t in batch_ids]
        return batch

    def _adjust_cpu(self) -> None:
        # Check system load average
        # 0: 1 min, 1: 5min, 2: 15 min
        #load_avg = psutil.getloadavg()[0]
        # use cpu
        if self.n_cpus == -1:
            cpu: float = psutil.cpu_percent()
        else:
            cpu: float = sum(psutil.cpu_percent(percpu=True)[:self.n_cpus]) / self.n_cpus

        if cpu >= 1.0:  # psutil returns 0.0 sometimes for good reasons
            self.latest = time.time()
            old_batch_size = self.current_batch_size
            if cpu >= self.target_cpu + 5:
                # Reduce batch size if system load is too high
                self.current_batch_size = min(self.max_batch, max(self.min_batch, int(self.current_batch_size * 0.9)))
            else:
                # Increase batch size if system load is low
                self.current_batch_size = max(self.min_batch, min(self.max_batch, int(self.current_batch_size * 1.5)))
            self.current_batch_size = int(self.current_batch_size)
            if self.current_batch_size != old_batch_size:
                tqdm.write(f"CPU: {cpu}, batch size: {self.current_batch_size}")
        else:
            tqdm.write(f"CPU: {cpu}, batch size: {self.current_batch_size} (UNCHANGED)")
