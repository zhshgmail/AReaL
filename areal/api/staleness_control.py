"""Staleness control strategies for different training modes.

This module contains the business logic for controlling sample staleness in
the rollout workflow. Different training modes (standard PPO, segment-wise PPO)
implement different staleness control policies.
"""

from __future__ import annotations

import queue
import traceback
from typing import TYPE_CHECKING, Any, Dict, List

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from areal.api.cache_api import RolloutCache
    from areal.api.cli_args import InferenceEngineConfig
    from areal.api.engine_api import InferenceEngine
    from areal.api.queue_api import RolloutQueue

RECOMPUTE_VERSION_KEY = "_recompute_version"


def _ensure_recompute_key(td: TensorDict) -> None:
    """Ensure the recompute tracking key exists with batch-aligned shape."""
    if not isinstance(td, TensorDict):
        return
    if RECOMPUTE_VERSION_KEY in td.keys():
        return
    if "versions" not in td.keys():
        return
    versions = td.get("versions")
    default_value = torch.full_like(versions[:, :1], -1, dtype=torch.int64)
    td.set(RECOMPUTE_VERSION_KEY, default_value)


class StalenessControlStrategy:
    """Base strategy for controlling staleness filtering behavior.

    This follows the Strategy pattern to encapsulate different staleness control
    policies. Strategies contain the actual business logic for checking, filtering,
    and purging stale samples.
    """

    def __init__(self, config: "InferenceEngineConfig" | None = None):
        """Initialize strategy with configuration.

        Args:
            config: Training configuration (optional for base strategy)
        """
        self.config = config

    def calculate_staleness(
        self,
        versions: List[int],
        loss_mask: List[int],
        current_ver: int,
        recompute_version: int,
        config: "InferenceEngineConfig",
    ) -> tuple[int, int, int]:
        """Calculate staleness metrics for a sample.

        Args:
            versions: Version list from sample
            loss_mask: Loss mask list from sample
            current_ver: Current model version
            recompute_version: Recompute version if sample was recomputed
            config: Training configuration

        Returns:
            Tuple of (staleness, allow_staleness, max_version)
        """
        output_positions = [idx for idx, mask in enumerate(loss_mask) if mask]
        output_versions = [versions[idx] for idx in output_positions if versions[idx] >= 0]

        if not output_versions:
            return (0, 1, -1)

        max_version = max(output_versions)
        min_version = min(output_versions)
        recomputed = recompute_version >= 0

        tail_staleness = current_ver - max_version
        head_staleness = current_ver - min_version if recomputed else None
        staleness = head_staleness if recomputed else tail_staleness

        allow_staleness = 1
        if recomputed and config.max_head_offpolicyness is not None:
            allow_staleness = max(allow_staleness, int(config.max_head_offpolicyness))

        return (staleness, allow_staleness, max_version)

    def is_sample_too_stale(
        self,
        td: TensorDict,
        current_ver: int,
        config: "InferenceEngineConfig",
    ) -> bool:
        """Check if a sample exceeds staleness threshold.

        Args:
            td: TensorDict sample to check
            current_ver: Current model version
            config: Training configuration

        Returns:
            True if sample should be dropped due to staleness
        """
        return False  # Default: no staleness check

    def purge_stale_samples_from_queue(
        self,
        output_queue: "RolloutQueue",
        current_ver: int,
        last_purged_ver: int,
        inference_engine: "InferenceEngine",
        result_cache: "RolloutCache",
        config: "InferenceEngineConfig",
        logger: Any,
    ) -> int:
        """Drain output queue and drop stale samples when version increases.

        Args:
            output_queue: Queue containing rollout outputs
            current_ver: Current model version
            last_purged_ver: Last version at which purge occurred
            inference_engine: Inference engine instance
            result_cache: Result cache for statistics
            config: Training configuration
            logger: Logger instance

        Returns:
            New last_purged_ver value
        """
        return last_purged_ver  # Default: no purging

    def filter_stale_from_cache(
        self,
        result_cache: "RolloutCache",
        current_ver: int,
        config: "InferenceEngineConfig",
        logger: Any,
    ) -> int:
        """Remove stale samples from result_cache.

        Args:
            result_cache: Cache to filter (modified in place)
            current_ver: Current model version
            config: Training configuration
            logger: Logger instance

        Returns:
            Number of samples dropped
        """
        return 0  # Default: no filtering

    def should_filter_before_enqueue(self) -> bool:
        """Whether to filter stale samples before enqueueing in rollout thread."""
        return False


class StandardPPOStrategy(StalenessControlStrategy):
    """Strategy for standard PPO - no staleness filtering.

    This provides backward-compatible behavior matching the original AReaL
    implementation before segment-wise PPO was added.
    """

    def __init__(self, config: "InferenceEngineConfig" | None = None):
        """Initialize standard PPO strategy.

        Args:
            config: Training configuration
        """
        super().__init__(config)


class SegmentWisePPOStrategy(StalenessControlStrategy):
    """Strategy for segment-wise decoupled PPO - full staleness control.

    Implements aggressive staleness filtering to ensure training stability:
    1. Purge stale samples from queue when version increases
    2. Filter stale samples from cache before returning
    3. Pre-filter samples before enqueue to prevent queue overflow
    """

    def __init__(self, config: "InferenceEngineConfig" | None = None):
        """Initialize segment-wise PPO strategy.

        Args:
            config: Training configuration
        """
        super().__init__(config)

    def is_sample_too_stale(
        self,
        td: TensorDict,
        current_ver: int,
        config: "InferenceEngineConfig",
    ) -> bool:
        """Check if a sample exceeds staleness threshold.

        Args:
            td: TensorDict sample to check
            current_ver: Current model version
            config: Training configuration

        Returns:
            True if sample should be dropped due to staleness
        """
        try:
            versions = td.get("versions", None)
            loss_mask = td.get("loss_mask", None)
            if versions is None or loss_mask is None:
                return False  # Keep samples without version info

            ver = versions[0].tolist()
            lm_row = loss_mask[0]
            if torch.is_tensor(lm_row):
                lm = lm_row.tolist()
            else:
                lm = list(lm_row)

            recompute_ver = self._extract_version(td.get(RECOMPUTE_VERSION_KEY, None))
            staleness, allow_staleness, _ = self.calculate_staleness(
                ver, lm, current_ver, recompute_ver or -1, config
            )

            return staleness > allow_staleness
        except Exception:
            traceback.print_exc()
            return False  # Keep on error

    def purge_stale_samples_from_queue(
        self,
        output_queue: "RolloutQueue",
        current_ver: int,
        last_purged_ver: int,
        inference_engine: "InferenceEngine",
        result_cache: "RolloutCache",
        config: "InferenceEngineConfig",
        logger: Any,
    ) -> int:
        """Drain output queue and drop stale samples when version increases.

        Args:
            output_queue: Queue containing rollout outputs
            current_ver: Current model version
            last_purged_ver: Last version at which purge occurred
            inference_engine: Inference engine instance
            result_cache: Result cache for statistics
            config: Training configuration
            logger: Logger instance

        Returns:
            New last_purged_ver value
        """
        if current_ver <= last_purged_ver:
            return last_purged_ver  # Only purge when version increases

        drained = 0
        picked_prev = 0
        dropped = 0
        kept = 0
        v0_picked = 0
        v0_kept = 0
        v0_dropped = 0
        put_back_buf: List[TensorDict] = []

        # Drain entire queue
        while True:
            try:
                traj = output_queue.get_nowait()
            except queue.Empty:
                break

            drained += 1
            try:
                if self.is_sample_too_stale(traj, current_ver, config):
                    dropped += 1
                    # Track v0 statistics
                    versions = traj.get("versions", None)
                    if versions is not None:
                        ver = versions[0].tolist()
                        loss_mask = traj.get("loss_mask", None)
                        if loss_mask is not None:
                            lm = loss_mask[0].tolist() if torch.is_tensor(loss_mask[0]) else list(loss_mask[0])
                            output_positions = [idx for idx, mask in enumerate(lm) if mask]
                            if output_positions:
                                max_version = max([ver[idx] for idx in output_positions if ver[idx] >= 0], default=-1)
                                if max_version == 0:
                                    v0_dropped += 1
                else:
                    put_back_buf.append(traj)
                    kept += 1
                    # Track statistics for kept samples
                    versions = traj.get("versions", None)
                    if versions is not None:
                        ver = versions[0].tolist()
                        loss_mask = traj.get("loss_mask", None)
                        if loss_mask is not None:
                            lm = loss_mask[0].tolist() if torch.is_tensor(loss_mask[0]) else list(loss_mask[0])
                            output_positions = [idx for idx, mask in enumerate(lm) if mask]
                            output_versions = [ver[idx] for idx in output_positions if ver[idx] >= 0]
                            if output_versions:
                                max_version = max(output_versions)
                                if max_version == 0:
                                    v0_kept += 1
                                if (current_ver - 1) in output_versions:
                                    picked_prev += 1
                                    if max_version == 0:
                                        v0_picked += 1
            except Exception:
                put_back_buf.append(traj)  # Keep on error
                traceback.print_exc()

        # Put items back with timeout to handle full queue
        for item in put_back_buf:
            try:
                _ensure_recompute_key(item)
                output_queue.put(item, timeout=1.0)
            except queue.Full:
                logger.error(
                    f"Output queue remains full after version purge. "
                    f"Queue size: {output_queue.qsize()}/{output_queue.maxsize}. "
                    f"Please increase queue_size."
                )
                raise RuntimeError(
                    "Output queue full when putting back after version purge. Please increase queue_size."
                )

        logger.info(
            f"[QueuePurge] ver_switch to {current_ver}: drained={drained} "
            f"picked_prev={picked_prev} dropped={dropped} kept={kept} "
            f"cache_size={result_cache.size()} "
            f"v0(picked/dropped/kept)={v0_picked}/{v0_dropped}/{v0_kept}"
        )
        return current_ver

    def filter_stale_from_cache(
        self,
        result_cache: "RolloutCache",
        current_ver: int,
        config: "InferenceEngineConfig",
        logger: Any,
    ) -> int:
        """Remove stale samples from result_cache.

        Args:
            result_cache: Cache to filter (modified in place)
            current_ver: Current model version
            config: Training configuration
            logger: Logger instance

        Returns:
            Number of samples dropped
        """
        # Use cache's filter_inplace method with predicate
        # Predicate returns True for items to KEEP (not stale)
        dropped_cache = result_cache.filter_inplace(
            lambda td: not self.is_sample_too_stale(td, current_ver, config)
        )

        if dropped_cache:
            logger.info(f"[CacheFilter] dropped_cache={dropped_cache} size={result_cache.size()}")

        return dropped_cache

    def should_filter_before_enqueue(self) -> bool:
        """Whether to filter stale samples before enqueueing in rollout thread."""
        return True

    @staticmethod
    def _extract_version(value: Any) -> int | None:
        """Extract version number from tensor or scalar."""
        if value is None:
            return None
        try:
            tensor = torch.as_tensor(value)
            tensor = tensor.reshape(-1)
            if tensor.numel() == 0:
                return None
            return int(tensor[0].item())
        except Exception:
            try:
                return int(value)
            except Exception:
                return None
