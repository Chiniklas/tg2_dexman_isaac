# DexSafeDagger

DexSafeDagger acquires a privileged teacher policy from RL trained in Isaac Sim and Isaac Lab. We then distill the teacher policies into an RGB-based student policy in an online SafeDagger-style setting.

## Installation
**Note**: This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

1. [Install](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) Isaac Sim and Isaac Lab by following the local Conda installation route.

**Note**: After cloning the Isaac Lab repository and before installing it, check out tag `v2.2.1` (it can also work with `v2.0.2` with minor code changes):
```bash
        cd <IsaacLab>
        git checkout v2.2.1
```
2. Install DEXTRAH for Isaac Lab in your new Conda environment.
```bash
        cd <DEXTRAH>
        python -m pip install -e .
```
4. Ensure that a sufficiently recent `GLIBCXX_` version can be found.
```bash
        conda install -c conda-forge libstdcxx-ng
        conda install -c conda-forge libgcc-ng=12 libstdcxx-ng=12
```

## DexSafeDagger Teacher Training
```bash
python train.py \
    --task=dextrah_tg2_inspirehand \
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
python play_test.py \
  --task dextrah_tg2_inspirehand \
  --num_envs 8 \
  --objects_dir test_object \
  --max_pose_angle 90 \
  --checkpoint /home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/rl_games/logs/rl_games/dextrah_lstm/2026-01-21_08-57-56/nn/dextrah_lstm.pth
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
