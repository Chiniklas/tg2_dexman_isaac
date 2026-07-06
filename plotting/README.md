# Plotting pipeline

TensorBoard ingestion and curve preprocessing are separate from rendering.
Export the enabled ablation runs in `config.yaml` once:

```bash
conda run -n dexsafedagger python plotting/export_ablation_raw_data.py \
  --config plotting/config.yaml \
  --output plotting/plots/<timestamp>/raw_data
```

Every timestamped plotting run is self-contained. Its `raw_data/manifest.json`
indexes one subdirectory per configured ablation run. Only tags used by the
paper plots are extracted. Smoothing and downsampling happen before saving, so
rendering does not reopen or reprocess the full TensorBoard series.

Each compact array has the named columns `step`, `value`, `wall_time`,
`smoothed_value`, and `band`. For example:

```python
import numpy as np

data = np.load(
    "plotting/plots/<timestamp>/raw_data/<ablation-run>/"
    "train%2Favg%2Funsafe_episode_rate.npy",
    allow_pickle=False,
)
steps = data["step"]
values = data["smoothed_value"]
```

Render plots from the saved data without opening TensorBoard event files:

```bash
python plotting/plot_multi_training_curve.py \
  --config plotting/config.yaml \
  --raw-data-dir plotting/plots/<timestamp>/raw_data
```

`run_all_plots.sh` performs the preprocessing export first and then passes the
timestamp-local dataset to both rendering stages. The plots and the compact
source arrays therefore remain together under the same timestamped directory.

Persistent compact-data saving is enabled by default and can be controlled by
the pipeline parser:

```bash
# Keep <timestamp>/raw_data (default).
plotting/run_all_plots.sh --save-raw-data true

# Use temporary compact data and remove it after rendering.
plotting/run_all_plots.sh --save-raw-data false
```
