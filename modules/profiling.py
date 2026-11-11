"""
Training pipeline profiler for diagnosing GPU utilization bottlenecks.

Measures dataloader wait time vs GPU compute time to quantify data starvation.
Integrates with TensorBoard for visualization.

Usage in training loop:
    profiler = TrainingProfiler()
    profiler.start_epoch()

    for batch_idx, batch_data in enumerate(dataloader):
        profiler.record_dataloader_wait()

        # ... move data to GPU, prepare tensors ...

        profiler.record_gpu_start()
        loss = model(data)
        loss.backward()
        optimizer.step()
        profiler.record_gpu_end()

        profiler.mark_batch_end()

    stats = profiler.get_epoch_stats()
    # Log stats['dataloader_wait_s'], stats['gpu_compute_s'], stats['ratio']
"""

import time

import torch


class TrainingProfiler:
    """
    Profiles the training loop to measure data loading vs GPU compute time.

    Tracks two key metrics across an epoch:
      - dataloader_wait_time: time spent waiting for the DataLoader to yield the next batch
      - gpu_compute_time:    time spent in forward + backward + optimizer step

    The ratio dataloader_wait / gpu_compute indicates the severity of data starvation:
      - ratio < 1.0:  GPU is the bottleneck (ideal)
      - ratio 1-3x:   mild data starvation
      - ratio > 3x:    significant data starvation — data loading needs optimization
    """

    def __init__(self):
        self.dataloader_wait_time = 0.0
        self.gpu_compute_time = 0.0
        self._t_batch_start = None
        self._first_batch = True
        self._batch_count = 0
        # (start_event, end_event) pairs, timed once at epoch end.
        # CUDA events avoid the per-step torch.cuda.synchronize() calls that
        # otherwise serialize CPU and GPU on every training step.
        self._gpu_events = []
        self._pending_start = None

    def start_epoch(self):
        """Reset accumulators at the start of each epoch."""
        self.dataloader_wait_time = 0.0
        self.gpu_compute_time = 0.0
        self._t_batch_start = time.time()
        self._first_batch = True
        self._batch_count = 0
        self._gpu_events = []

    def record_dataloader_wait(self):
        """
        Call at the TOP of the batch loop, immediately after receiving batch_data
        from the DataLoader. Measures how long we waited since the previous batch
        finished processing.
        """
        t_now = time.time()
        if not self._first_batch:
            self.dataloader_wait_time += t_now - self._t_batch_start
        self._first_batch = False
        self._batch_count += 1

    def record_gpu_start(self):
        """
        Call right BEFORE the GPU computation (forward/backward/step).
        Records a CUDA event (non-blocking) as the start timestamp.
        """
        if torch.cuda.is_available():
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._pending_start = ev
        else:
            self._pending_start = None

    def record_gpu_end(self):
        """
        Call right AFTER the GPU computation (forward/backward/step).
        Records a CUDA event (non-blocking); elapsed time is accumulated
        lazily in get_epoch_stats() after a single epoch-end synchronize.
        """
        if self._pending_start is not None and self._batch_count > 1:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._gpu_events.append((self._pending_start, ev))
        self._pending_start = None

    def mark_batch_end(self):
        """
        Call at the END of batch processing (after logging, scheduler step, etc.).
        Sets the timestamp from which the next dataloader wait will be measured.
        """
        self._t_batch_start = time.time()

    def get_epoch_stats(self):
        """
        Return profiling statistics for the completed epoch.

        Returns:
            dict with keys:
                dataloader_wait_s:  total seconds spent waiting for DataLoader
                gpu_compute_s:      total seconds spent on GPU computation
                ratio:              wait / compute (higher = worse data starvation)
                batches:            number of batches profiled
        """
        # Single synchronize per epoch to read all event pairs
        if self._gpu_events:
            torch.cuda.synchronize()
            self.gpu_compute_time = sum(
                s.elapsed_time(e) for s, e in self._gpu_events
            ) / 1000.0  # ms -> s

        if self.gpu_compute_time > 0:
            ratio = self.dataloader_wait_time / self.gpu_compute_time
        else:
            ratio = float('inf')

        return {
            'dataloader_wait_s': self.dataloader_wait_time,
            'gpu_compute_s': self.gpu_compute_time,
            'ratio': ratio,
            'batches': self._batch_count,
        }

    def log_epoch_stats(self, logger, epoch, tb_writer=None):
        """
        Log epoch profiling stats to logger and optionally TensorBoard.

        Args:
            logger: logging.Logger instance
            epoch: current epoch number
            tb_writer: optional SummaryWriter for TensorBoard
        """
        stats = self.get_epoch_stats()
        logger.info(
            f"[PROFILE] Epoch {epoch}: "
            f"dataloader_wait={stats['dataloader_wait_s']:.1f}s "
            f"gpu_compute={stats['gpu_compute_s']:.1f}s "
            f"wait/compute_ratio={stats['ratio']:.1f}x "
            f"(batches={stats['batches']})"
        )
        if stats['ratio'] > 3.0:
            logger.warning(
                f"[PROFILE] Epoch {epoch}: Significant data starvation detected "
                f"(ratio={stats['ratio']:.1f}x > 3x). "
                f"Consider: (1) converting NC to NPZ format, "
                f"(2) reducing num_workers, "
                f"(3) enabling persistent_workers"
            )

        if tb_writer is not None:
            tb_writer.add_scalar('Profile/Dataloader_Wait_Sec', stats['dataloader_wait_s'], epoch)
            tb_writer.add_scalar('Profile/GPU_Compute_Sec', stats['gpu_compute_s'], epoch)
            tb_writer.add_scalar('Profile/Wait_Compute_Ratio', stats['ratio'], epoch)

        return stats


class IOTimer:
    """
    Lightweight context manager for measuring I/O operation latency.

    Usage:
        with IOTimer("wave_nc_load") as t:
            data = load_nc_file(path)
        print(f"Loaded in {t.elapsed_ms:.1f}ms")
    """

    def __init__(self, label=""):
        self.label = label
        self.elapsed_s = 0.0
        self._t_start = None

    @property
    def elapsed_ms(self):
        return self.elapsed_s * 1000

    def __enter__(self):
        self._t_start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_s = time.time() - self._t_start


def enable_gpu_optimizations():
    """
    Enable GPU performance optimizations for Ampere+ architectures (A100, H100, L20X, etc.).

    - TF32: ~2x matmul throughput with minimal precision loss
    - cuDNN benchmark: auto-tunes convolution algorithms for fastest path

    Should be called once per process, after CUDA device initialization.
    Safe to call multiple times — subsequent calls are no-ops.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
