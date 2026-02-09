import torch


class OODGaussianBuffer:
    def __init__(self, config=None, default_obs_key=None, rank=0):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.obs_key = cfg.get("obs_key", None)
        self.default_obs_key = default_obs_key
        self.buffer_size = int(cfg.get("buffer_size", 100_000))
        self.min_samples = int(cfg.get("min_samples", 5_000))
        self.update_interval = int(cfg.get("update_interval", 1_000))
        self.threshold_quantile = float(cfg.get("threshold_quantile", 0.99))
        self.diag_eps = float(cfg.get("diag_eps", 1e-4))
        self.rank = rank

        self.initialized = False
        self.buffer = None
        self.buf_idx = 0
        self.buf_count = 0
        self.steps = 0
        self.mean = None
        self.var = None
        self.threshold = None

    def set_default_obs_key(self, key):
        self.default_obs_key = key

    def init_buffer(self, obs):
        if not self.enabled or self.initialized:
            return
        feats = self._extract_features(obs)
        feat_dim = feats.reshape(feats.shape[0], -1).shape[1]
        self.buffer = torch.empty(
            (self.buffer_size, feat_dim), dtype=torch.float32, device="cpu"
        )
        self.buf_idx = 0
        self.buf_count = 0
        self.steps = 0
        self.mean = None
        self.var = None
        self.threshold = None
        self.initialized = True
        if self.rank == 0:
            print(
                f"OOD enabled: buffer_size={self.buffer_size}, "
                f"min_samples={self.min_samples}, update_interval={self.update_interval}, "
                f"quantile={self.threshold_quantile}"
            )

    def check_ood(self, obs, device):
        if not self.enabled:
            return None
        if not self.initialized:
            self.init_buffer(obs)
        feats = self._extract_features(obs)
        self._update_buffer(feats)
        self.steps += 1
        if self.steps % self.update_interval == 0:
            self._refit_stats()
        if self.buf_count < self.min_samples or self.threshold is None:
            return torch.ones(feats.shape[0], dtype=torch.bool, device=device)
        feats_cpu = feats.detach().to("cpu").reshape(feats.shape[0], -1)
        scores = self._score_from_stats(feats_cpu, self.mean, self.var)
        unsafe = scores > self.threshold
        return unsafe.to(device)

    def _extract_features(self, obs):
        if obs is None:
            raise ValueError("obs is required for OOD feature extraction.")
        key = self.obs_key or self.default_obs_key
        if key is None:
            raise ValueError("OOD obs_key is not set and no default_obs_key is available.")
        if key not in obs:
            raise KeyError(f"OOD obs_key '{key}' not found in obs.")
        feats = obs[key]
        return feats.detach().to(dtype=torch.float32)

    def _update_buffer(self, feats):
        feats_cpu = feats.detach().to("cpu").reshape(feats.shape[0], -1)
        num = feats_cpu.shape[0]
        if num >= self.buffer_size:
            feats_cpu = feats_cpu[-self.buffer_size:]
            num = feats_cpu.shape[0]
        end = self.buf_idx + num
        if end <= self.buffer_size:
            self.buffer[self.buf_idx:end] = feats_cpu
        else:
            first = self.buffer_size - self.buf_idx
            self.buffer[self.buf_idx:] = feats_cpu[:first]
            self.buffer[: end % self.buffer_size] = feats_cpu[first:]
        self.buf_idx = end % self.buffer_size
        self.buf_count = min(self.buffer_size, self.buf_count + num)

    def _refit_stats(self):
        if self.buf_count < self.min_samples:
            return
        data = self.buffer[: self.buf_count]
        mean = data.mean(dim=0)
        var = data.var(dim=0, unbiased=False) + self.diag_eps
        self.mean = mean
        self.var = var
        scores = self._score_from_stats(data, mean, var)
        self.threshold = torch.quantile(scores, self.threshold_quantile).item()

    def _score_from_stats(self, feats, mean, var):
        diff = feats - mean
        return 0.5 * ((diff * diff) / var + torch.log(var)).sum(dim=-1)
