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
git lfs pull --include="dextrah_lab/assets/**"
```

The simulator needs the real USD/STL files, not Git LFS pointer files. A quick check should show zero pointer files under `dextrah_lab/assets`:
```bash
rg -l "version https://git-lfs.github.com/spec/v1" dextrah_lab/assets | wc -l
```

3. Install DEXTRAH runtime dependencies.
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
mkdir -p dextrah_lab/assets/test_object
cp -r dextrah_lab/assets/test_objects/USD dextrah_lab/assets/test_object/
```

## SimToolReal TG2 Teacher Training
```bash
python dextrah_lab/train_rl_games.py \
  --task simtoolreal_tg2 \
  --agent_cfg rl_games_sapo_cfg.yaml \
  --num_envs 1024 \
  --headless
```

## Replay SimToolReal TG2 Policy
```bash
python dextrah_lab/play_rl_games.py \
  --task simtoolreal_tg2 \
  --checkpoint dextrah_lab/tasks/simtoolreal_tg2/logs/<run>/nn/<checkpoint>.pth
```

## Camera-based Student Distillation
**Note**: Before starting student training, download the visual texture data (`textures.zip`) and place its contents inside the `dextrah_lab/assets` directory. Download the assets from this [link](https://huggingface.co/datasets/nvidia/dextrah_textures/blob/main/textures.zip) and unzip them into the assets folder.

1. Ablation 1: Vanilla DAgger

```bash
python /home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/distillation_new/run_distillation_safedagger.py \
  --task=dextrah_tg2_inspirehand \
  --num_envs 32 \
  --enable_cameras \
  --teacher multi_object_distillation \
  --unsafe_mode none \
  --eval_every 2500 \
  --eval_num_episodes 3 \
  env.distillation=True \
  env.simulate_stereo=True \
  env.objects_dir=distill_multi_objects \
  env.enable_adr=False
```

2. Ablation 2: Vanilla SafeDagger
```bash
python /home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/distillation_new/run_distillation_safedagger.py \
  --pipeline safedagger \
  --task dextrah_tg2_inspirehand \
  --num_envs 32 \
  --enable_cameras \
  --teacher multi_object_distillation \
  --unsafe_mode l2 \
  --eval_every 2500 \
  --eval_num_episodes 3 \
  env.distillation=True \
  env.simulate_stereo=True \
  env.objects_dir=distill_multi_objects \
  env.enable_adr=False
```

3. Ablation 3: SafeDagger with Predictor
```bash
python run_distillation_safedagger.py \
  --pipeline warmstart \
  --task dextrah_tg2_inspirehand \
  --num_envs 32 \
  --headless \
  --enable_cameras \
  --teacher multi_object_distillation \
  --unsafe_mode failure_predictor \
  --failure_predictor_type critic \
  env.distillation=True \
  env.simulate_stereo=True \
  env.objects_dir=distill_multi_objects \
  env.enable_adr=False
```

## Replay Student Policy
```bash
cd /home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/distillation_new
python eval.py \
  --task=dextrah_tg2_inspirehand \
  --num_envs 8 \
  --enable_cameras \
  --checkpoint /home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/distillation_new/dextrah_student_safe_dagger.pth \
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
  --task dextrah_tg2_inspirehand \
  --num_envs 32 \
  --eval_episodes 10 \
  --teacher_policy_dir /home/chizhang/projects/dextrah/tg2_dexman_isaac/pretrained_ckpts/teacher_eval \
  --teacher_object_dir /home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/assets/teacher_eval
```

2. Student policy evaluation
```bash
python /home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/distillation_new/eval_student.py \
  --headless \
  --enable_cameras \
  --task dextrah_tg2_inspirehand \
  --checkpoint /home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/distillation_new/runs/dextrah-tg2-inspirehand-safedagger_24-05-44-09/nn/dextrah_student_safe_dagger.pth.pth \
  --objects_dir distill_multi_objects \
  --num_envs 32 \
  --num_episodes 3 \
  --file_name_head student_eval_metrics \
  --metrics_output_json /home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/distillation_new/eval_results/student_eval_metrics.json \
  env.enable_adr=False \
  env.distillation=True \
  env.simulate_stereo=True
```
