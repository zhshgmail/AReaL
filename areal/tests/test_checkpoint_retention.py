"""Unit tests for checkpoint retention management."""

import os
import tempfile
from pathlib import Path

import pytest

from areal.api.cli_args import SaverConfig
from areal.api.io_struct import FinetuneSpec
from areal.utils.checkpoint_retention import (
    ArchiveAction,
    CheckpointInfo,
    CheckpointRetentionManager,
    CheckpointType,
    CompressArchiveAction,
    DeleteAction,
    RetentionPolicy,
)
from areal.utils.saver import Saver


class TestCheckpointInfo:
    """Test suite for CheckpointInfo dataclass."""

    def test_from_path_parsing(self):
        """Test parsing checkpoint info from directory path."""
        path = "/tmp/checkpoints/default/epoch5epochstep99globalstep599"
        info = CheckpointInfo.from_path(path, CheckpointType.STEP)

        assert info is not None
        assert info.epoch == 5
        assert info.step == 99
        assert info.global_step == 599
        assert info.checkpoint_type == CheckpointType.STEP
        assert info.path == path

    def test_from_path_invalid(self):
        """Test parsing invalid checkpoint path returns None."""
        invalid_path = "/tmp/checkpoints/invalid_name"
        info = CheckpointInfo.from_path(invalid_path, CheckpointType.STEP)
        assert info is None

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        original = CheckpointInfo(
            path="/tmp/ckpt",
            epoch=1,
            step=10,
            global_step=100,
            checkpoint_type=CheckpointType.EPOCH,
            timestamp=1234567890.0,
        )

        data = original.to_dict()
        restored = CheckpointInfo.from_dict(data)

        assert restored.path == original.path
        assert restored.epoch == original.epoch
        assert restored.step == original.step
        assert restored.global_step == original.global_step
        assert restored.checkpoint_type == original.checkpoint_type
        assert restored.timestamp == original.timestamp


class TestDeleteAction:
    """Test suite for DeleteAction cleanup strategy."""

    def test_delete_existing_directory(self):
        """Test deleting an existing checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "checkpoint")
            os.makedirs(ckpt_path)
            Path(os.path.join(ckpt_path, "model.safetensors")).touch()

            info = CheckpointInfo(
                path=ckpt_path,
                epoch=0,
                step=0,
                global_step=0,
                checkpoint_type=CheckpointType.STEP,
                timestamp=0.0,
            )

            action = DeleteAction()
            result = action.execute(info)

            assert result is True
            assert not os.path.exists(ckpt_path)

    def test_delete_nonexistent_directory(self):
        """Test deleting a non-existent directory returns False."""
        info = CheckpointInfo(
            path="/nonexistent/path",
            epoch=0,
            step=0,
            global_step=0,
            checkpoint_type=CheckpointType.STEP,
            timestamp=0.0,
        )

        action = DeleteAction()
        result = action.execute(info)
        assert result is False


class TestArchiveAction:
    """Test suite for ArchiveAction cleanup strategy."""

    def test_archive_checkpoint(self):
        """Test archiving a checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_root = os.path.join(tmpdir, "archive")
            ckpt_path = os.path.join(tmpdir, "checkpoint")
            os.makedirs(ckpt_path)
            Path(os.path.join(ckpt_path, "model.safetensors")).touch()

            info = CheckpointInfo(
                path=ckpt_path,
                epoch=0,
                step=0,
                global_step=0,
                checkpoint_type=CheckpointType.STEP,
                timestamp=0.0,
            )

            action = ArchiveAction(archive_root)
            result = action.execute(info)

            assert result is True
            assert not os.path.exists(ckpt_path)
            archived_path = os.path.join(archive_root, "checkpoint")
            assert os.path.exists(archived_path)
            assert os.path.exists(os.path.join(archived_path, "model.safetensors"))


class TestCompressArchiveAction:
    """Test suite for CompressArchiveAction cleanup strategy."""

    def test_compress_and_archive_checkpoint(self):
        """Test compressing and archiving a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_root = os.path.join(tmpdir, "archive")
            ckpt_path = os.path.join(tmpdir, "checkpoint")
            os.makedirs(ckpt_path)
            Path(os.path.join(ckpt_path, "model.safetensors")).touch()

            info = CheckpointInfo(
                path=ckpt_path,
                epoch=0,
                step=0,
                global_step=0,
                checkpoint_type=CheckpointType.STEP,
                timestamp=0.0,
            )

            action = CompressArchiveAction(archive_root, compression="gz")
            result = action.execute(info)

            assert result is True
            assert not os.path.exists(ckpt_path)
            archived_path = os.path.join(archive_root, "checkpoint.tar.gz")
            assert os.path.exists(archived_path)


class TestRetentionPolicy:
    """Test suite for RetentionPolicy configuration."""

    def test_should_cleanup_with_max_to_keep(self):
        """Test cleanup trigger based on max_to_keep."""
        policy = RetentionPolicy(max_to_keep=3)

        assert not policy.should_cleanup(2)
        assert not policy.should_cleanup(3)
        assert policy.should_cleanup(4)
        assert policy.should_cleanup(5)

    def test_should_not_cleanup_when_protected(self):
        """Test protected policy never triggers cleanup."""
        policy = RetentionPolicy(max_to_keep=3, protect=True)
        assert not policy.should_cleanup(100)

    def test_should_not_cleanup_when_unlimited(self):
        """Test unlimited retention never triggers cleanup."""
        policy = RetentionPolicy(max_to_keep=None)
        assert not policy.should_cleanup(100)

    def test_get_num_to_remove(self):
        """Test calculation of number of checkpoints to remove."""
        policy = RetentionPolicy(max_to_keep=3)

        assert policy.get_num_to_remove(2) == 0
        assert policy.get_num_to_remove(3) == 0
        assert policy.get_num_to_remove(4) == 1
        assert policy.get_num_to_remove(5) == 2
        assert policy.get_num_to_remove(10) == 7

    def test_create_delete_action(self):
        """Test creating delete cleanup action."""
        policy = RetentionPolicy(cleanup_action="delete")
        action = policy.create_cleanup_action()
        assert isinstance(action, DeleteAction)

    def test_create_archive_action(self):
        """Test creating archive cleanup action."""
        policy = RetentionPolicy(cleanup_action="archive", archive_root="/tmp/archive")
        action = policy.create_cleanup_action()
        assert isinstance(action, ArchiveAction)

    def test_create_compress_archive_action(self):
        """Test creating compress archive cleanup action."""
        policy = RetentionPolicy(
            cleanup_action="compress_archive", archive_root="/tmp/archive"
        )
        action = policy.create_cleanup_action()
        assert isinstance(action, CompressArchiveAction)

    def test_archive_action_requires_archive_root(self):
        """Test that archive actions require archive_root."""
        policy = RetentionPolicy(cleanup_action="archive", archive_root=None)
        with pytest.raises(ValueError, match="archive_root must be specified"):
            policy.create_cleanup_action()


class TestCheckpointRetentionManager:
    """Test suite for CheckpointRetentionManager."""

    def test_register_and_cleanup_step_checkpoints(self):
        """Test registering step checkpoints and automatic cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_save_root = os.path.join(tmpdir, "checkpoints")
            os.makedirs(model_save_root)

            step_policy = RetentionPolicy(max_to_keep=2, cleanup_action="delete")
            manager = CheckpointRetentionManager(
                model_save_root=model_save_root, step_policy=step_policy
            )

            # Create and register 5 step checkpoints
            checkpoints = []
            for i in range(5):
                ckpt_path = os.path.join(
                    model_save_root, f"epoch0epochstep{i}globalstep{i}"
                )
                os.makedirs(ckpt_path)
                Path(os.path.join(ckpt_path, "model.safetensors")).touch()
                checkpoints.append(ckpt_path)

                manager.register_checkpoint(
                    checkpoint_path=ckpt_path,
                    epoch=0,
                    step=i,
                    global_step=i,
                    is_epoch_checkpoint=False,
                )

            # Only last 2 should exist
            assert not os.path.exists(checkpoints[0])
            assert not os.path.exists(checkpoints[1])
            assert not os.path.exists(checkpoints[2])
            assert os.path.exists(checkpoints[3])
            assert os.path.exists(checkpoints[4])

    def test_protected_epoch_checkpoints(self):
        """Test that epoch checkpoints are protected when policy says so."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_save_root = os.path.join(tmpdir, "checkpoints")
            os.makedirs(model_save_root)

            epoch_policy = RetentionPolicy(
                max_to_keep=1, cleanup_action="delete", protect=True
            )
            manager = CheckpointRetentionManager(
                model_save_root=model_save_root, epoch_policy=epoch_policy
            )

            # Create and register 5 epoch checkpoints
            checkpoints = []
            for i in range(5):
                ckpt_path = os.path.join(
                    model_save_root, f"epoch{i}epochstep99globalstep{i * 100 + 99}"
                )
                os.makedirs(ckpt_path)
                Path(os.path.join(ckpt_path, "model.safetensors")).touch()
                checkpoints.append(ckpt_path)

                manager.register_checkpoint(
                    checkpoint_path=ckpt_path,
                    epoch=i,
                    step=99,
                    global_step=i * 100 + 99,
                    is_epoch_checkpoint=True,
                )

            # All epoch checkpoints should still exist because protected
            for ckpt_path in checkpoints:
                assert os.path.exists(ckpt_path)

    def test_separate_retention_for_epoch_and_step(self):
        """Test that epoch and step checkpoints have independent retention."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_save_root = os.path.join(tmpdir, "checkpoints")
            os.makedirs(model_save_root)

            epoch_policy = RetentionPolicy(
                max_to_keep=2, cleanup_action="delete", protect=False
            )
            step_policy = RetentionPolicy(max_to_keep=3, cleanup_action="delete")
            manager = CheckpointRetentionManager(
                model_save_root=model_save_root,
                epoch_policy=epoch_policy,
                step_policy=step_policy,
            )

            # Create 5 epoch checkpoints
            epoch_checkpoints = []
            for i in range(5):
                ckpt_path = os.path.join(
                    model_save_root, f"epoch{i}epochstep99globalstep{i * 100 + 99}"
                )
                os.makedirs(ckpt_path)
                Path(os.path.join(ckpt_path, "model.safetensors")).touch()
                epoch_checkpoints.append(ckpt_path)

                manager.register_checkpoint(
                    checkpoint_path=ckpt_path,
                    epoch=i,
                    step=99,
                    global_step=i * 100 + 99,
                    is_epoch_checkpoint=True,
                )

            # Create 5 step checkpoints
            step_checkpoints = []
            for i in range(5):
                ckpt_path = os.path.join(
                    model_save_root, f"epoch0epochstep{i}globalstep{i}"
                )
                os.makedirs(ckpt_path)
                Path(os.path.join(ckpt_path, "model.safetensors")).touch()
                step_checkpoints.append(ckpt_path)

                manager.register_checkpoint(
                    checkpoint_path=ckpt_path,
                    epoch=0,
                    step=i,
                    global_step=i,
                    is_epoch_checkpoint=False,
                )

            # Check epoch checkpoints: keep last 2
            assert not os.path.exists(epoch_checkpoints[0])
            assert not os.path.exists(epoch_checkpoints[1])
            assert not os.path.exists(epoch_checkpoints[2])
            assert os.path.exists(epoch_checkpoints[3])
            assert os.path.exists(epoch_checkpoints[4])

            # Check step checkpoints: keep last 3
            assert not os.path.exists(step_checkpoints[0])
            assert not os.path.exists(step_checkpoints[1])
            assert os.path.exists(step_checkpoints[2])
            assert os.path.exists(step_checkpoints[3])
            assert os.path.exists(step_checkpoints[4])

    def test_scan_and_register_existing_checkpoints(self):
        """Test scanning and registering existing checkpoints for migration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_save_root = os.path.join(tmpdir, "checkpoints")
            os.makedirs(model_save_root)

            # Create some existing checkpoints without metadata
            os.makedirs(os.path.join(model_save_root, "epoch0epochstep50globalstep50"))
            os.makedirs(os.path.join(model_save_root, "epoch0epochstep99globalstep99"))
            os.makedirs(os.path.join(model_save_root, "epoch1epochstep50globalstep150"))

            # Create recovery checkpoint (should be ignored)
            os.makedirs(os.path.join(model_save_root, "recover_checkpoint"))

            manager = CheckpointRetentionManager(model_save_root=model_save_root)
            manager.scan_and_register_existing_checkpoints(steps_per_epoch=100)

            # Load metadata and verify
            metadata = manager._load_metadata()
            assert len(metadata) == 3

            # Check that checkpoints were correctly classified
            step_checkpoints = [
                info
                for info in metadata.values()
                if info.checkpoint_type == CheckpointType.STEP
            ]
            epoch_checkpoints = [
                info
                for info in metadata.values()
                if info.checkpoint_type == CheckpointType.EPOCH
            ]

            assert len(step_checkpoints) == 2
            assert len(epoch_checkpoints) == 1

    def test_metadata_persistence(self):
        """Test that metadata is correctly saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_save_root = os.path.join(tmpdir, "checkpoints")
            os.makedirs(model_save_root)

            manager = CheckpointRetentionManager(model_save_root=model_save_root)

            # Register a checkpoint
            ckpt_path = os.path.join(model_save_root, "epoch0epochstep0globalstep0")
            os.makedirs(ckpt_path)

            manager.register_checkpoint(
                checkpoint_path=ckpt_path,
                epoch=0,
                step=0,
                global_step=0,
                is_epoch_checkpoint=False,
            )

            # Create new manager instance and verify it can load metadata
            manager2 = CheckpointRetentionManager(model_save_root=model_save_root)
            metadata = manager2._load_metadata()

            assert len(metadata) == 1
            assert ckpt_path in metadata
            assert metadata[ckpt_path].epoch == 0
            assert metadata[ckpt_path].step == 0
            assert metadata[ckpt_path].global_step == 0


class TestSaverIntegration:
    """Test suite for Saver integration with retention management."""

    def test_saver_with_retention_disabled(self):
        """Test that Saver works normally when retention is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SaverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                enable_retention=False,
            )
            ft_spec = FinetuneSpec(
                total_train_epochs=1, dataset_size=100, train_batch_size=10
            )

            saver = Saver(config, ft_spec)
            manager = saver._get_retention_manager()

            assert manager is None

    def test_saver_with_retention_enabled(self):
        """Test that Saver creates retention manager when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SaverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                enable_retention=True,
                step_max_to_keep=2,
            )
            ft_spec = FinetuneSpec(
                total_train_epochs=1, dataset_size=100, train_batch_size=10
            )

            saver = Saver(config, ft_spec)
            manager = saver._get_retention_manager()

            assert manager is not None
            assert isinstance(manager, CheckpointRetentionManager)

    def test_saver_default_archive_root(self):
        """Test that Saver creates default archive root path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SaverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                enable_retention=True,
                step_max_to_keep=2,
                step_cleanup_action="archive",
                archive_root=None,
            )
            ft_spec = FinetuneSpec(
                total_train_epochs=1, dataset_size=100, train_batch_size=10
            )

            saver = Saver(config, ft_spec)
            manager = saver._get_retention_manager()

            assert manager is not None
            assert manager.step_policy.archive_root is not None
            assert "checkpoints_archive" in manager.step_policy.archive_root

    def test_backward_compatibility_scan(self):
        """Test that existing checkpoints are scanned on first use."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint directory structure
            model_save_root = Saver.get_model_save_root(
                "test_exp", "test_trial", tmpdir, "default"
            )
            os.makedirs(os.path.join(model_save_root, "epoch0epochstep0globalstep0"))

            config = SaverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                enable_retention=True,
            )
            ft_spec = FinetuneSpec(
                total_train_epochs=1, dataset_size=100, train_batch_size=10
            )

            saver = Saver(config, ft_spec)
            manager = saver._get_retention_manager()

            # Verify existing checkpoint was registered
            metadata = manager._load_metadata()
            assert len(metadata) > 0
