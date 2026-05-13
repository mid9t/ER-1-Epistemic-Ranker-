from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA


class GDADensityModel:
    """
    Density-Aware EDL density estimator.

    Pipeline:
    CLS embedding -> PCA whitening -> per-class Gaussian densities
    -> prior-weighted marginal log-density -> sigmoid density score.
    """

    def __init__(self, n_components: int = 64, temperature: float = 0.1):
        self.n_components = n_components
        self.temperature = temperature
        self.pca = None
        self.gmm = None
        self.class_means = None
        self.class_covs = None
        self.log_priors = None
        self.train_mu = None
        self.train_sigma = None
        self.jitter_eps = None
        self.device = torch.device("cpu")
        self.fitted = False
        self.pca_components_gpu = None
        self.pca_mean_gpu = None
        self.pca_scale_gpu = None

    @staticmethod
    def _gda_dtype(target_device: torch.device) -> torch.dtype:
        return torch.float32 if target_device.type == "mps" else torch.double

    def to(self, target_device):
        target_device = torch.device(target_device)
        target_dtype = self._gda_dtype(target_device)

        if self.class_means is not None:
            self.class_means = self.class_means.to(device=target_device, dtype=target_dtype)
        if self.class_covs is not None:
            self.class_covs = self.class_covs.to(device=target_device, dtype=target_dtype)
        if self.log_priors is not None:
            self.log_priors = self.log_priors.to(device=target_device, dtype=target_dtype)
        if self.train_mu is not None:
            self.train_mu = self.train_mu.to(device=target_device, dtype=target_dtype)
        if self.train_sigma is not None:
            self.train_sigma = self.train_sigma.to(device=target_device, dtype=target_dtype)

        if self.pca_components_gpu is not None:
            self.pca_components_gpu = self.pca_components_gpu.to(
                device=target_device, dtype=target_dtype
            )
        if self.pca_mean_gpu is not None:
            self.pca_mean_gpu = self.pca_mean_gpu.to(device=target_device, dtype=target_dtype)
        if self.pca_scale_gpu is not None:
            self.pca_scale_gpu = self.pca_scale_gpu.to(device=target_device, dtype=target_dtype)

        if self.class_means is not None and self.class_covs is not None:
            jitter = self.jitter_eps * torch.eye(
                self.n_components,
                dtype=self.class_covs.dtype,
                device=target_device,
            ).unsqueeze(0)
            self.gmm = torch.distributions.MultivariateNormal(
                loc=self.class_means,
                covariance_matrix=self.class_covs + jitter,
            )

        self.device = target_device
        return self

    def fit(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        print(
            f"\n[GDA] Fitting on {len(embeddings):,} embeddings "
            f"(PCA {embeddings.shape[1]}->{self.n_components})..."
        )

        emb_np = embeddings.cpu().float().numpy()
        labels = labels.cpu()
        self.pca = PCA(n_components=self.n_components, whiten=True, random_state=42)
        emb_pca = self.pca.fit_transform(emb_np)
        emb_t = torch.tensor(emb_pca, dtype=torch.double)

        explained = self.pca.explained_variance_ratio_.sum()
        print(f"  PCA variance explained: {explained:.1%}")

        num_classes = int(labels.max().item()) + 1
        class_counts = torch.zeros(num_classes, dtype=torch.double)
        means, covs = [], []

        for class_idx in range(num_classes):
            mask = labels == class_idx
            count = mask.sum().item()
            if count < 2:
                raise ValueError(f"Class {class_idx} needs at least two samples for QDA.")
            class_counts[class_idx] = count
            cls_emb = emb_t[mask]
            mu_c = cls_emb.mean(dim=0)
            centered = cls_emb - mu_c
            cov_c = (centered.T @ centered) / (count - 1)
            means.append(mu_c)
            covs.append(cov_c)
            print(f"  Class {class_idx}: {count:,} samples")

        self.log_priors = torch.log(class_counts / class_counts.sum())
        print(f"  Log-priors: {[f'{p.item():.3f}' for p in self.log_priors]}")

        means_t = torch.stack(means)
        covs_t = torch.stack(covs)
        gmm = None
        double_info = torch.finfo(torch.double)
        jitters = [0, double_info.tiny] + [10**exp for exp in range(-308, 0, 1)]

        for jitter_eps in jitters:
            try:
                jitter = jitter_eps * torch.eye(self.n_components, dtype=torch.double).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=means_t,
                    covariance_matrix=covs_t + jitter,
                )
                break
            except (RuntimeError, ValueError) as exc:
                message = str(exc).lower()
                if "cholesky" in message or "covariance" in message:
                    continue
                raise

        if gmm is None:
            raise RuntimeError("[GDA] Covariance not positive definite at max jitter.")

        self.class_means = means_t
        self.class_covs = covs_t
        self.jitter_eps = jitter_eps
        self.gmm = gmm
        print(f"  QDA fit OK (jitter={jitter_eps:.2e})")

        self._cache_pca_gpu(self.device)

        log_dens = self._raw_log_density(emb_t.to(dtype=self.class_means.dtype))
        self.train_mu = log_dens.mean()
        self.train_sigma = log_dens.std().clamp_min(1e-6)
        print(f"  Log-density stats: mu={self.train_mu:.2f}, sigma={self.train_sigma:.2f}")

        self.fitted = True

    def _cache_pca_gpu(self, target_device):
        device = torch.device(target_device)
        dtype = self._gda_dtype(device)
        self.pca_components_gpu = torch.tensor(self.pca.components_, dtype=dtype, device=device)
        self.pca_mean_gpu = torch.tensor(self.pca.mean_, dtype=dtype, device=device)
        if self.pca.whiten:
            self.pca_scale_gpu = torch.tensor(
                np.sqrt(self.pca.explained_variance_),
                dtype=dtype,
                device=device,
            )
        else:
            self.pca_scale_gpu = None
        print(f"  PCA cached on {device} ({dtype})")

    def _pca_transform_gpu(self, embeddings: torch.Tensor) -> torch.Tensor:
        assert self.pca_components_gpu is not None, "Call fit() before PCA transform."
        if embeddings.device != self.pca_components_gpu.device:
            embeddings = embeddings.to(self.pca_components_gpu.device)
        x = embeddings.to(self.pca_components_gpu.dtype) - self.pca_mean_gpu
        x = x @ self.pca_components_gpu.T
        if self.pca_scale_gpu is not None:
            x = x / self.pca_scale_gpu
        return x.to(self.class_means.dtype)

    def _raw_log_density(self, emb_pca: torch.Tensor) -> torch.Tensor:
        log_cls = self.gmm.log_prob(emb_pca.unsqueeze(1))
        log_prior = self.log_priors.to(log_cls.device)
        weighted = log_cls + log_prior.unsqueeze(0)
        return torch.logsumexp(weighted, dim=-1)

    def score(self, embeddings: torch.Tensor) -> torch.Tensor:
        assert self.fitted, "Call fit() before score()."
        with torch.no_grad():
            emb_pca = self._pca_transform_gpu(embeddings)
            log_p = self._raw_log_density(emb_pca)
            mu = self.train_mu.to(log_p.device)
            sigma = self.train_sigma.to(log_p.device)
            z = (log_p - mu) / (sigma * self.temperature)
            return torch.sigmoid(z).float()

    def density_histogram(self, embeddings: torch.Tensor, pair_types: list, title: str = "") -> dict:
        assert self.fitted, "Call fit() before density_histogram()."
        s = self.score(embeddings)
        results = {}

        for pair_type in ("positive", "hard_negative", "easy_negative", "ood"):
            idx = [i for i, p in enumerate(pair_types) if p == pair_type]
            if idx:
                results[pair_type] = round(s[idx].mean().item(), 4)

        if title:
            print(f"\n[GDA Density] {title}")
            for pair_type, value in results.items():
                if pair_type == "ood" and value < 0.3:
                    tag = " <- good"
                elif pair_type != "ood" and value > 0.3:
                    tag = " <- in-dist"
                else:
                    tag = ""
                print(f"  {pair_type.capitalize():>15}: s(h)={value:.4f}{tag}")

        return results
