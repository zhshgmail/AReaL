# MODEL_FAMILY specifies how the pretrained checkpoint is loaded, e.g., as a LLaMA model or a GPT model.
MODEL_FAMILY=qwen2

# PRETRAINED_PATH is the HuggingFace checkpoint.
PRETRAINED_PATH=/storage/openpsi/models/Qwen__Qwen2.5-7B-Instruct
TRAIN_DATA_PATH=/storage/openpsi/data/ppu_test_data/examples_test_data/sft_pos-train.jsonl
VALID_DATA_PATH=/storage/openpsi/data/ppu_test_data/examples_test_data/sft_pos-train.jsonl

# Option 1: The experiment runs locally with subprocesses.
# MODE=local
# Option 2: The experiment runs in a Ray cluster
# MODE=ray
# Option 3: The experiment runs in a SLURM + pyxis cluster
# Using the slurm mode requires a cluster spec file
# and setting CLUSTER_SPEC_PATH to the path of it.
MODE=slurm

# `experiment_name` and `trial_name` can be arbitrary.
# Logs and saved checkpoints will be indexed by them.
EXP_NAME=quickstart-sft
TRIAL_NAME=$MODEL_FAMILY-$MODE-run1

# We use the "manual" allocation mode here to manually specify the parallelism strategy,
# which is pipeline=2, tensor-model=2, and data=2, using in total of 8 GPUs.

# The `sft` subcommand specifies that this is a supervised fine-tuning experiment.
export CLUSTER_SPEC_PATH="/storage/realhf/examples/cluster_config.json"
python3 -m realhf.apps.quickstart sft \
    mode=$MODE \
    experiment_name=$EXP_NAME \
    trial_name=$TRIAL_NAME \
    exp_ctrl.total_train_epochs=8 \
    exp_ctrl.save_freq_steps=50 \
    exp_ctrl.eval_freq_epochs=1 \
    model.optimizer.type=adam \
    model.optimizer.lr_scheduler_type=cosine \
    model.optimizer.lr=1e-5 \
    model.optimizer.warmup_steps_proportion=0.02 \
    model.type._class=$MODEL_FAMILY \
    model.path=$PRETRAINED_PATH \
    dataset.train_path=${TRAIN_DATA_PATH} \
    dataset.valid_path=${VALID_DATA_PATH} \
    dataset.max_seqlen=1024 \
    dataset.train_bs_n_seqs=512 \
    dataset.valid_bs_n_seqs=512 \
    allocation_mode=d4m4p2 \
    n_nodes=4 n_gpus_per_node=8 \
    allocation.mb_spec.n_mbs=2