# DexSafeDagger

DexSafeDagger acquires a privileged teacher policy from RL trained in Isaac Sim and Isaac Lab. We then distill the teacher policies into an RGB-based student policy in an online SafeDagger-style setting.

## Installation
**Note**: This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

1. Create the `dexsafedagger` Conda environment, then [install](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) Isaac Sim `5.0.0` and Isaac Lab `2.2.1` into that environment by following the local Conda installation route.

```bash
conda create -n dexsafedagger python=3.11 -y
conda activate dexsafedagger
```

**Note**: After cloning the Isaac Lab repository and before installing it, check out tag `v2.2.1` (it can also work with `v2.0.2` with minor code changes):
```bash
cd <IsaacLab>
git checkout v2.2.1
```

2. Activate the `dexsafedagger` Conda environment from Step 1, then clone this repository with Git LFS assets.
```bash
conda activate dexsafedagger
cd ~/projects
git clone git@github.com:Chiniklas/tg2_dexman_isaac.git
cd tg2_dexman_isaac
git lfs install
git lfs pull --include="dexsafedagger_lab/assets/**"
```

The simulator needs the real USD/STL files, not Git LFS pointer files. A quick check should show zero pointer files under `dexsafedagger_lab/assets`:
```bash
rg -l "version https://git-lfs.github.com/spec/v1" dexsafedagger_lab/assets | wc -l
```

3. Install DEXSAFEDAGGER runtime dependencies.
```bash
./install_runtime_deps.sh
```

The helper script runs `python -m pip install -e .`, installs `urdfpy==0.0.22` without its stale `networkx==2.2` dependency pin, restores Isaac-compatible pins, and verifies that the TG2 URDF can be loaded. Plain `pip install -e .` installs the package metadata, but it cannot safely express this `urdfpy` workaround.

4. Ensure that a sufficiently recent `GLIBCXX_` version can be found.
```bash
conda install -c conda-forge libstdcxx-ng
conda install -c conda-forge libgcc-ng=12 libstdcxx-ng=12
```

5. If you use the small checked-in test objects, make sure they are in the object-directory layout expected by the task:
```bash
mkdir -p dexsafedagger_lab/assets/test_object
cp -r dexsafedagger_lab/assets/test_objects/USD dexsafedagger_lab/assets/test_object/
```

## DexSafeDagger Teacher Training
```bash
cd dexsafedagger_lab/rl_games
python train.py \
    --task=dexsafedagger_tg2_inspirehand \
    --seed 42 \
    --num_envs 128 \
    --headless \
    agent.params.config.minibatch_size=256 \
    agent.params.config.central_value_config.minibatch_size=256 \
    agent.params.config.learning_rate=0.0001 \
    agent.params.config.horizon_length=16 \
    agent.params.config.mini_epochs=4 \
    agent.params.config.multi_gpu=False \
    agent.wandb_activate=False \
    env.success_for_adr=0.4 \
    env.objects_dir=test_object \
    env.use_cuda_graph=False
```

## Replay Teacher Policy
```bash
python play.py \
  --task dexsafedagger_tg2_inspirehand \
  --num_envs 8 \
  --objects_dir test_object \
  --max_pose_angle 90 \
  --checkpoint /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/rl_games/logs/rl_games/dexsafedagger_lstm/2026-01-21_08-57-56/nn/dexsafedagger_lstm.pth
```

## Camera-based Student Distillation
**Note**: Before starting student training, download the visual texture data (`textures.zip`) and place its contents inside the `dexsafedagger_lab/assets` directory. Download the assets from this [link](https://huggingface.co/datasets/nvidia/dexsafedagger_textures/blob/main/textures.zip) and unzip them into the assets folder.

1. Ablation 1: Vanilla DAgger

```bash
python /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/distillation/scripts/run_distillation_safedagger.py \
  --variant vanilla_dagger \
  --task=dexsafedagger_tg2_inspirehand \
  --num_envs 32 \
  --enable_cameras \
  --teacher multi_object_distillation \
  --eval_every 2500 \
  --eval_num_episodes 3 \
  env.distillation=True \
  env.simulate_stereo=True \
  env.objects_dir=distill_multi_objects \
  env.enable_adr=False
```

2. Ablation 2: Vanilla SafeDagger
```bash
python /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/distillation/scripts/run_distillation_safedagger.py \
  --variant vanilla_safedagger \
  --task dexsafedagger_tg2_inspirehand \
  --num_envs 32 \
  --enable_cameras \
  --teacher multi_object_distillation \
  --eval_every 2500 \
  --eval_num_episodes 3 \
  env.distillation=True \
  env.simulate_stereo=True \
  env.objects_dir=distill_multi_objects \
  env.enable_adr=False
```

3. Ablation 3: DexSafeDagger
```bash
python /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/distillation/scripts/run_distillation_safedagger.py \
  --variant dexsafedagger \
  --task dexsafedagger_tg2_inspirehand \
  --num_envs 32 \
  --headless \
  --enable_cameras \
  --teacher multi_object_distillation \
  env.distillation=True \
  env.simulate_stereo=True \
  env.objects_dir=distill_multi_objects \
  env.enable_adr=False
```

4. Experimental Ablation: DexSafeDaggerUltra

This is currently a brainstorming scaffold, not a runnable training method. The
idea is to use a VLM to predict teacher intervention points from visual/context
observations instead of relying on a fixed threshold. The code exposes the
variant and config structure, but the VLM backend still needs to be implemented.

```bash
python /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/distillation/scripts/run_distillation_safedagger.py \
  --variant dexsafedaggerUltra \
  --task dexsafedagger_tg2_inspirehand \
  --num_envs 32 \
  --headless \
  --enable_cameras \
  --teacher multi_object_distillation \
  env.distillation=True \
  env.simulate_stereo=True \
  env.objects_dir=distill_multi_objects \
  env.enable_adr=False
```

## Replay Student Policy
```bash
cd /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/distillation
python scripts/replay.py \
  --task=dexsafedagger_tg2_inspirehand \
  --num_envs 8 \
  --enable_cameras \
  --checkpoint /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/distillation/dexsafedagger_student_safe_dagger.pth \
  --num_episodes 10 \
  env.distillation=True \
  env.simulate_stereo=True \
  env.objects_dir=distill_multi_objects
```

Optional recording flags:
```bash
  --record_data --max_records_per_file 100 --create_video
```

## Evaluation
1. Teacher policy evaluation
```bash
python3 eval.py \
  --headless \
  --task dexsafedagger_tg2_inspirehand \
  --num_envs 32 \
  --eval_episodes 10 \
  --teacher_policy_dir /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/pretrained_ckpts/teacher_eval \
  --teacher_object_dir /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/assets/teacher_eval
```

2. Student policy evaluation
```bash
python /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/distillation/eval/eval_student.py \
  --headless \
  --enable_cameras \
  --task dexsafedagger_tg2_inspirehand \
  --checkpoint /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/distillation/runs/dexsafedagger-tg2-inspirehand-safedagger_24-05-44-09/nn/dexsafedagger_student_safe_dagger.pth.pth \
  --objects_dir distill_multi_objects \
  --num_envs 32 \
  --num_episodes 3 \
  --file_name_head student_eval_metrics \
  --metrics_output_json /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/distillation/eval_results/student_eval_metrics.json \
  env.enable_adr=False \
  env.distillation=True \
  env.simulate_stereo=True
```
