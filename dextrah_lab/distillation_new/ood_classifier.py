import torch
import torch.nn as nn


class OODGaussianBuffer:
    def __init__(self, config=None, default_obs_key=None, rank=0):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.obs_key = cfg.get("obs_key", None)
        self.default_obs_key = default_obs_key
        self.min_samples = int(cfg.get("min_samples", 5_000))
        self.update_interval = int(cfg.get("update_interval", 1_000))
        self.threshold_quantile = float(cfg.get("threshold_quantile", 0.75))
        self.diag_eps = float(cfg.get("diag_eps", 1e-4))
        self.rank = rank

        self.initialized = False
        self.steps = 0
        self.mean = None
        self.var = None
        self.threshold = None
        # Welford running stats (global, no forgetting)
        self._count = 0
        self._mean = None
        self._m2 = None
        self._p2 = P2Quantile(self.threshold_quantile)

    def set_default_obs_key(self, key):
        self.default_obs_key = key

    def init_buffer(self, obs):
        if not self.enabled or self.initialized:
            return
        feats = self._extract_features(obs)
        self.steps = 0
        self.mean = None
        self.var = None
        self.threshold = None
        self._count = 0
        self._mean = None
        self._m2 = None
        self._p2 = P2Quantile(self.threshold_quantile)
        self.initialized = True
        if self.rank == 0:
            print(
                f"OOD enabled: min_samples={self.min_samples}, "
                f"update_interval={self.update_interval}, quantile={self.threshold_quantile}"
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
        if self._count < self.min_samples or self.threshold is None:
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
        self._update_welford(feats_cpu)
        self._update_p2(feats_cpu)

    def _refit_stats(self):
        if self._count < self.min_samples:
            return
        mean = self._mean
        var = (self._m2 / max(self._count - 1, 1)) + self.diag_eps
        self.mean = mean
        self.var = var
        if self._p2 is None or not self._p2.ready or self._count < self.min_samples:
            return
        self.threshold = self._p2.value

    def _score_from_stats(self, feats, mean, var):
        diff = feats - mean
        return 0.5 * ((diff * diff) / var + torch.log(var)).sum(dim=-1)

    def _update_welford(self, feats_cpu):
        # feats_cpu: (N, D) on CPU
        if feats_cpu.numel() == 0:
            return
        batch = feats_cpu.to(dtype=torch.float32)
        bsz = batch.shape[0]
        if self._count == 0:
            self._count = bsz
            self._mean = batch.mean(dim=0)
            diff = batch - self._mean
            self._m2 = (diff * diff).sum(dim=0)
            return
        batch_mean = batch.mean(dim=0)
        diff = batch - batch_mean
        batch_m2 = (diff * diff).sum(dim=0)
        total = self._count + bsz
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * (bsz / total)
        self._m2 = self._m2 + batch_m2 + (delta * delta) * (self._count * bsz / total)
        self._count = total

    def _update_p2(self, feats_cpu):
        if self._p2 is None:
            return
        if self._mean is None or self._m2 is None or self._count < 2:
            return
        mean = self._mean
        var = (self._m2 / max(self._count - 1, 1)) + self.diag_eps
        scores = self._score_from_stats(feats_cpu, mean, var)
        for s in scores.tolist():
            self._p2.add(float(s))


class P2Quantile:
    def __init__(self, q):
        if not (0.0 < q < 1.0):
            raise ValueError(f"q must be in (0, 1), got {q}")
        self.q = float(q)
        self._init = []
        self.n = None
        self.np = None
        self.dn = None
        self.qv = None
        self.ready = False
        self.value = None

    def add(self, x):
        if not self.ready:
            self._init.append(float(x))
            if len(self._init) == 5:
                self._init.sort()
                self.qv = list(self._init)
                self.n = [1, 2, 3, 4, 5]
                p = self.q
                self.np = [
                    1.0,
                    1.0 + 2.0 * p,
                    1.0 + 4.0 * p,
                    3.0 + 2.0 * p,
                    5.0,
                ]
                self.dn = [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0]
                self.ready = True
                self.value = self.qv[2]
            return

        x = float(x)
        if x < self.qv[0]:
            self.qv[0] = x
            k = 0
        elif x < self.qv[1]:
            k = 0
        elif x < self.qv[2]:
            k = 1
        elif x < self.qv[3]:
            k = 2
        elif x <= self.qv[4]:
            k = 3
        else:
            self.qv[4] = x
            k = 3

        for i in range(k + 1, 5):
            self.n[i] += 1
        for i in range(5):
            self.np[i] += self.dn[i]

        for i in range(1, 4):
            d = self.np[i] - self.n[i]
            if (d >= 1.0 and self.n[i + 1] - self.n[i] > 1) or (
                d <= -1.0 and self.n[i - 1] - self.n[i] < -1
            ):
                d_sign = 1 if d >= 0 else -1
                qhat = self.qv[i] + d_sign / (self.n[i + 1] - self.n[i - 1]) * (
                    (self.n[i] - self.n[i - 1] + d_sign)
                    * (self.qv[i + 1] - self.qv[i])
                    / (self.n[i + 1] - self.n[i])
                    + (self.n[i + 1] - self.n[i] - d_sign)
                    * (self.qv[i] - self.qv[i - 1])
                    / (self.n[i] - self.n[i - 1])
                )
                if self.qv[i - 1] < qhat < self.qv[i + 1]:
                    self.qv[i] = qhat
                else:
                    self.qv[i] = self.qv[i] + d_sign * (
                        (self.qv[i + d_sign] - self.qv[i]) / (self.n[i + d_sign] - self.n[i])
                    )
                self.n[i] += d_sign

        self.value = self.qv[2]


class OODPCABuffer:
    def __init__(self, config=None, default_obs_key=None, rank=0):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.obs_key = cfg.get("obs_key", None)
        self.default_obs_key = default_obs_key
        self.buffer_size = int(cfg.get("buffer_size", 100_000))
        self.min_samples = int(cfg.get("min_samples", 5_000))
        self.update_interval = int(cfg.get("update_interval", 1_000))
        self.threshold_quantile = float(cfg.get("threshold_quantile", 0.99))
        self.pca_dim = int(cfg.get("pca_dim", 32))
        self.score_type = str(cfg.get("score_type", "reconstruction")).lower()
        self.diag_eps = float(cfg.get("diag_eps", 1e-4))
        self.rank = rank

        if self.score_type not in {"reconstruction", "mahalanobis"}:
            raise ValueError(f"Unsupported PCA score_type: {self.score_type}")

        self.initialized = False
        self.buffer = None
        self.buf_idx = 0
        self.buf_count = 0
        self.steps = 0
        self.mean = None
        self.components = None
        self.eigvals = None
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
        self.components = None
        self.eigvals = None
        self.threshold = None
        self.initialized = True
        if self.rank == 0:
            print(
                "OOD PCA enabled: buffer_size={}, min_samples={}, update_interval={}, "
                "quantile={}, pca_dim={}, score_type={}".format(
                    self.buffer_size,
                    self.min_samples,
                    self.update_interval,
                    self.threshold_quantile,
                    self.pca_dim,
                    self.score_type,
                )
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
            self._refit_pca()
        if self.buf_count < self.min_samples or self.threshold is None:
            return torch.ones(feats.shape[0], dtype=torch.bool, device=device)
        feats_cpu = feats.detach().to("cpu").reshape(feats.shape[0], -1)
        scores = self._score_from_pca(feats_cpu)
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

    def _refit_pca(self):
        if self.buf_count < self.min_samples:
            return
        data = self.buffer[: self.buf_count]
        mean = data.mean(dim=0)
        x = data - mean
        n, d = x.shape
        k = min(self.pca_dim, n, d)
        if k <= 0:
            return
        # Low-rank PCA on CPU; x is already centered.
        _, s, v = torch.pca_lowrank(x, q=k, center=False)
        eigvals = (s * s) / max(n - 1, 1)
        self.mean = mean
        self.components = v  # (d, k)
        self.eigvals = eigvals
        scores = self._score_from_pca(data)
        self.threshold = torch.quantile(scores, self.threshold_quantile).item()

    def _score_from_pca(self, feats):
        if self.mean is None or self.components is None:
            raise ValueError("PCA stats not fitted; call _refit_pca first.")
        x = feats - self.mean
        z = x @ self.components
        if self.score_type == "reconstruction":
            recon = z @ self.components.t()
            residual = x - recon
            return (residual * residual).sum(dim=-1)
        if self.score_type == "mahalanobis":
            denom = self.eigvals + self.diag_eps
            return (z * z / denom).sum(dim=-1)
        raise ValueError(f"Unsupported PCA score_type: {self.score_type}")


class OODMLPBuffer:
    def __init__(self, config=None, default_obs_key=None, rank=0):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.obs_key = cfg.get("obs_key", None)
        self.default_obs_key = default_obs_key
        self.buffer_size = int(cfg.get("buffer_size", 100_000))
        self.min_samples = int(cfg.get("min_samples", 5_000))
        self.update_interval = int(cfg.get("update_interval", 200))
        self.threshold_quantile = float(cfg.get("threshold_quantile", 0.95))
        self.hidden_sizes = self._parse_hidden_sizes(cfg.get("hidden_sizes", [256, 128]))
        self.lr = float(cfg.get("lr", 1e-3))
        self.weight_decay = float(cfg.get("weight_decay", 1e-6))
        self.batch_size = int(cfg.get("batch_size", 1024))
        self.train_steps = int(cfg.get("train_steps", 100))
        self.noise_std = float(cfg.get("noise_std", 0.1))
        self.threshold_samples = int(cfg.get("threshold_samples", 50_000))
        self.dropout_prob = float(cfg.get("dropout_prob", 0.1))
        self.device = str(cfg.get("device", "cpu")).lower()
        self.rank = rank

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            if self.rank == 0:
                print("OOD MLP: CUDA requested but not available; using CPU.")
        self.model_device = torch.device(self.device)
        if not (0.0 <= self.dropout_prob < 1.0):
            raise ValueError(f"dropout_prob must be in [0, 1), got {self.dropout_prob}")

        self.initialized = False
        self.buffer = None
        self.buf_idx = 0
        self.buf_count = 0
        self.steps = 0
        self.threshold = None
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.BCEWithLogitsLoss()
        # Notes for mixed proprio + image-embedding inputs:
        # - Prefer a pooled or CLS image embedding over raw tokens (already used in this setup).
        # - Consider LayerNorm on each modality before concatenation.
        # - Optional small projection heads (e.g., proprio->64, image->64) can stabilize scales.
        # - Light dropout (0.1-0.2) after concat is a cheap regularizer.

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
        self.threshold = None
        self._build_model(feat_dim)
        self.initialized = True
        if self.rank == 0:
            print(
                "OOD MLP enabled: buffer_size={}, min_samples={}, update_interval={}, "
                "quantile={}, hidden_sizes={}, lr={}, train_steps={}".format(
                    self.buffer_size,
                    self.min_samples,
                    self.update_interval,
                    self.threshold_quantile,
                    self.hidden_sizes,
                    self.lr,
                    self.train_steps,
                )
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
            self._train_classifier()
        if self.buf_count < self.min_samples or self.threshold is None:
            return torch.ones(feats.shape[0], dtype=torch.bool, device=device)
        feats_cpu = feats.detach().to("cpu").reshape(feats.shape[0], -1)
        scores = self._score_from_model(feats_cpu)
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

    def _build_model(self, input_dim):
        layers = []
        in_dim = input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if self.dropout_prob > 0.0:
                layers.append(nn.Dropout(p=self.dropout_prob))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers).to(self.model_device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def _train_classifier(self):
        if self.model is None or self.buffer is None:
            return
        if self.buf_count < self.min_samples:
            return
        data = self.buffer[: self.buf_count]
        bs = min(self.batch_size, self.buf_count)
        if bs <= 0:
            return
        self.model.train()
        for _ in range(self.train_steps):
            idx = torch.randint(0, self.buf_count, (bs,))
            pos = data[idx].to(self.model_device)
            neg = pos + self.noise_std * torch.randn_like(pos)
            x = torch.cat([pos, neg], dim=0)
            y = torch.cat(
                [
                    torch.ones(pos.shape[0], device=self.model_device),
                    torch.zeros(neg.shape[0], device=self.model_device),
                ],
                dim=0,
            )
            logits = self.model(x).squeeze(-1)
            loss = self.loss_fn(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self._refit_threshold()

    def _refit_threshold(self):
        if self.buffer is None or self.buf_count < self.min_samples:
            return
        data = self.buffer[: self.buf_count]
        if self.threshold_samples > 0 and data.shape[0] > self.threshold_samples:
            idx = torch.randint(0, data.shape[0], (self.threshold_samples,))
            data = data[idx]
        scores = self._score_from_model(data)
        self.threshold = torch.quantile(scores, self.threshold_quantile).item()

    def _score_from_model(self, feats):
        if self.model is None:
            raise ValueError("MLP model not initialized.")
        self.model.eval()
        x = feats.to(self.model_device)
        with torch.no_grad():
            logits = self.model(x).squeeze(-1)
            scores = 1.0 - torch.sigmoid(logits)
        return scores.detach().to("cpu")

    @staticmethod
    def _parse_hidden_sizes(value):
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip()]
            return [int(p) for p in parts]
        return [256, 128]
