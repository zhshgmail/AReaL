"""Checkpoint retention policies and cleanup strategies.

This module provides extensible checkpoint retention management with support for:
- Different retention policies for epoch-level vs step-level checkpoints
- Multiple cleanup actions (delete, archive, compress)
- Backward compatibility with existing recovery mechanisms
"""

import json
import os
import shutil
import tarfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from areal.utils import logging

logger = logging.getLogger("checkpoint_retention")


class CheckpointType(str, Enum):
    """Type of checkpoint for retention policy purposes."""

    EPOCH = "epoch"
    STEP = "step"
    RECOVERY = "recovery"


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint."""

    path: str
    epoch: int
    step: int
    global_step: int
    checkpoint_type: CheckpointType
    timestamp: float

    def to_dict(self):
        return {
            "path": self.path,
            "epoch": self.epoch,
            "step": self.step,
            "global_step": self.global_step,
            "checkpoint_type": self.checkpoint_type.value,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict):
        d = d.copy()
        d["checkpoint_type"] = CheckpointType(d["checkpoint_type"])
        return cls(**d)

    @classmethod
    def from_path(cls, path: str, checkpoint_type: CheckpointType):
        """Parse checkpoint info from directory name.

        Expected format: epoch{epoch}epochstep{step}globalstep{globalstep}
        """
        dirname = os.path.basename(path)
        try:
            parts = dirname.split("epochstep")
            epoch_part = parts[0].replace("epoch", "")
            step_globalstep = parts[1].split("globalstep")
            step = int(step_globalstep[0])
            global_step = int(step_globalstep[1])
            epoch = int(epoch_part)

            # Get directory creation time as timestamp
            timestamp = os.path.getctime(path) if os.path.exists(path) else 0.0

            return cls(
                path=path,
                epoch=epoch,
                step=step,
                global_step=global_step,
                checkpoint_type=checkpoint_type,
                timestamp=timestamp,
            )
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse checkpoint path {path}: {e}")
            return None


class CleanupAction(ABC):
    """Abstract base class for checkpoint cleanup actions."""

    @abstractmethod
    def execute(self, checkpoint_info: CheckpointInfo) -> bool:
        """Execute cleanup action on a checkpoint.

        Args:
            checkpoint_info: Information about the checkpoint to clean up.

        Returns:
            True if cleanup was successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this cleanup action."""
        pass


class DeleteAction(CleanupAction):
    """Delete checkpoint directory."""

    def execute(self, checkpoint_info: CheckpointInfo) -> bool:
        try:
            path = checkpoint_info.path
            if os.path.exists(path):
                shutil.rmtree(path)
                logger.info(f"Deleted checkpoint: {path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_info.path}: {e}")
            return False

    def get_name(self) -> str:
        return "delete"


class ArchiveAction(CleanupAction):
    """Move checkpoint to archive directory."""

    def __init__(self, archive_root: str):
        self.archive_root = archive_root
        os.makedirs(archive_root, exist_ok=True)

    def execute(self, checkpoint_info: CheckpointInfo) -> bool:
        try:
            src_path = checkpoint_info.path
            if not os.path.exists(src_path):
                return False

            # Create archive destination
            rel_path = os.path.basename(src_path)
            dst_path = os.path.join(self.archive_root, rel_path)

            # Move checkpoint to archive
            shutil.move(src_path, dst_path)
            logger.info(f"Archived checkpoint from {src_path} to {dst_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to archive checkpoint {checkpoint_info.path}: {e}")
            return False

    def get_name(self) -> str:
        return "archive"


class CompressArchiveAction(CleanupAction):
    """Compress and archive checkpoint to save space."""

    def __init__(self, archive_root: str, compression: str = "gz"):
        self.archive_root = archive_root
        self.compression = compression
        os.makedirs(archive_root, exist_ok=True)

    def execute(self, checkpoint_info: CheckpointInfo) -> bool:
        try:
            src_path = checkpoint_info.path
            if not os.path.exists(src_path):
                return False

            # Create compressed archive
            rel_path = os.path.basename(src_path)
            archive_name = f"{rel_path}.tar.{self.compression}"
            dst_path = os.path.join(self.archive_root, archive_name)

            # Compress checkpoint
            with tarfile.open(dst_path, f"w:{self.compression}") as tar:
                tar.add(src_path, arcname=rel_path)

            # Remove original after successful compression
            shutil.rmtree(src_path)
            logger.info(
                f"Compressed and archived checkpoint from {src_path} to {dst_path}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to compress and archive checkpoint {checkpoint_info.path}: {e}"
            )
            return False

    def get_name(self) -> str:
        return f"compress_archive_{self.compression}"


@dataclass
class RetentionPolicy:
    """Retention policy configuration for checkpoints."""

    max_to_keep: int | None = None
    cleanup_action: Literal["delete", "archive", "compress_archive"] = "delete"
    archive_root: str | None = None
    protect: bool = False

    def should_cleanup(self, num_checkpoints: int) -> bool:
        """Check if cleanup should be triggered based on current checkpoint count."""
        if self.protect:
            return False
        if self.max_to_keep is None:
            return False
        return num_checkpoints > self.max_to_keep

    def get_num_to_remove(self, num_checkpoints: int) -> int:
        """Calculate how many checkpoints should be removed."""
        if not self.should_cleanup(num_checkpoints):
            return 0
        return num_checkpoints - self.max_to_keep

    def create_cleanup_action(self) -> CleanupAction:
        """Create appropriate cleanup action based on configuration."""
        if self.cleanup_action == "delete":
            return DeleteAction()
        elif self.cleanup_action == "archive":
            if self.archive_root is None:
                raise ValueError("archive_root must be specified for archive action")
            return ArchiveAction(self.archive_root)
        elif self.cleanup_action == "compress_archive":
            if self.archive_root is None:
                raise ValueError(
                    "archive_root must be specified for compress_archive action"
                )
            return CompressArchiveAction(self.archive_root)
        else:
            raise ValueError(f"Unknown cleanup action: {self.cleanup_action}")


class CheckpointRetentionManager:
    """Manages checkpoint retention across different checkpoint types."""

    METADATA_FILE = ".checkpoint_metadata.json"

    def __init__(
        self,
        model_save_root: str,
        epoch_policy: RetentionPolicy | None = None,
        step_policy: RetentionPolicy | None = None,
    ):
        self.model_save_root = model_save_root
        self.epoch_policy = epoch_policy or RetentionPolicy(protect=True)
        self.step_policy = step_policy or RetentionPolicy()
        self.metadata_file = os.path.join(model_save_root, self.METADATA_FILE)

    def _load_metadata(self) -> dict[str, CheckpointInfo]:
        """Load checkpoint metadata from disk."""
        if not os.path.exists(self.metadata_file):
            return {}

        try:
            with open(self.metadata_file) as f:
                data = json.load(f)
                return {k: CheckpointInfo.from_dict(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Failed to load checkpoint metadata: {e}")
            return {}

    def _save_metadata(self, metadata: dict[str, CheckpointInfo]):
        """Save checkpoint metadata to disk."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump({k: v.to_dict() for k, v in metadata.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint metadata: {e}")

    def register_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        step: int,
        global_step: int,
        is_epoch_checkpoint: bool,
    ):
        """Register a new checkpoint and trigger cleanup if needed.

        Args:
            checkpoint_path: Path to the checkpoint directory.
            epoch: Training epoch number.
            step: Step within the epoch.
            global_step: Global training step.
            is_epoch_checkpoint: Whether this is an epoch-level checkpoint.
        """
        checkpoint_type = (
            CheckpointType.EPOCH if is_epoch_checkpoint else CheckpointType.STEP
        )

        # Create checkpoint info
        checkpoint_info = CheckpointInfo(
            path=checkpoint_path,
            epoch=epoch,
            step=step,
            global_step=global_step,
            checkpoint_type=checkpoint_type,
            timestamp=(
                os.path.getctime(checkpoint_path)
                if os.path.exists(checkpoint_path)
                else 0.0
            ),
        )

        # Load existing metadata
        metadata = self._load_metadata()

        # Add new checkpoint
        metadata[checkpoint_path] = checkpoint_info

        # Save updated metadata
        self._save_metadata(metadata)

        # Trigger cleanup for the checkpoint type
        self._cleanup_checkpoints(checkpoint_type)

    def _cleanup_checkpoints(self, checkpoint_type: CheckpointType):
        """Cleanup old checkpoints based on retention policy."""
        policy = (
            self.epoch_policy
            if checkpoint_type == CheckpointType.EPOCH
            else self.step_policy
        )

        if policy.protect or policy.max_to_keep is None:
            return

        # Load metadata
        metadata = self._load_metadata()

        # Filter checkpoints by type and remove those that don't exist
        checkpoints = [
            info
            for info in metadata.values()
            if info.checkpoint_type == checkpoint_type and os.path.exists(info.path)
        ]

        # Sort by global_step (oldest first)
        checkpoints.sort(key=lambda x: x.global_step)

        # Check if cleanup is needed
        num_to_remove = policy.get_num_to_remove(len(checkpoints))
        if num_to_remove <= 0:
            return

        # Get checkpoints to remove (oldest ones)
        checkpoints_to_remove = checkpoints[:num_to_remove]

        # Create cleanup action
        cleanup_action = policy.create_cleanup_action()

        # Execute cleanup
        for checkpoint_info in checkpoints_to_remove:
            if cleanup_action.execute(checkpoint_info):
                # Remove from metadata
                if checkpoint_info.path in metadata:
                    del metadata[checkpoint_info.path]

        # Save updated metadata
        self._save_metadata(metadata)

    def scan_and_register_existing_checkpoints(self, steps_per_epoch: int):
        """Scan checkpoint directory and register existing checkpoints.

        This is useful for migration from older versions that don't have metadata.

        Args:
            steps_per_epoch: Number of steps per epoch, used to determine if a
                checkpoint is epoch-level (step == steps_per_epoch - 1).
        """
        if not os.path.exists(self.model_save_root):
            return

        metadata = {}

        for entry in os.listdir(self.model_save_root):
            path = os.path.join(self.model_save_root, entry)

            # Skip metadata file and non-directories
            if entry == self.METADATA_FILE or not os.path.isdir(path):
                continue

            # Skip recovery checkpoint
            if entry == "recover_checkpoint" or entry == "recover_info":
                continue

            # Try to parse checkpoint info
            # Check if this is epoch checkpoint (last step of epoch)
            checkpoint_info = CheckpointInfo.from_path(path, CheckpointType.STEP)
            if checkpoint_info is None:
                continue

            # Determine if this is an epoch checkpoint
            is_epoch_checkpoint = checkpoint_info.step == steps_per_epoch - 1
            if is_epoch_checkpoint:
                checkpoint_info.checkpoint_type = CheckpointType.EPOCH

            metadata[path] = checkpoint_info

        # Save metadata
        if metadata:
            self._save_metadata(metadata)
            logger.info(f"Registered {len(metadata)} existing checkpoints")
