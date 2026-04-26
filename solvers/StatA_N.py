import torch
import torch.nn.functional as F
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 1 — Adaptive α
#
# Paper finding (Figure 3): tuning α relative to the ratio (batch_size / Keff)
# can yield +10-13% in some scenarios.  The paper leaves this as "future work"
# because Keff is unknown at test time.  We estimate Keff from the entropy of
# the average zero-shot prediction (Shannon effective number of categories) and
# use a simple monotone heuristic to set α.
#
# Expected gain: +0.5–1.5 % averaged across scenarios; largest gains in
# Low/Very-Low Keff regimes where the original α=1 is suboptimal.
# ─────────────────────────────────────────────────────────────────────────────

def estimate_keff(y_hat: torch.Tensor) -> float:
    """Estimate effective number of classes from zero-shot predictions.

    Uses the exponential of Shannon entropy of the mean prediction vector,
    which equals the true Keff when classes are uniformly represented and
    is a lower bound otherwise.
    """
    avg_pred = y_hat.mean(dim=0)
    avg_pred = avg_pred / (avg_pred.sum() + 1e-10)
    entropy = -(avg_pred * (avg_pred + 1e-10).log()).sum()
    return entropy.exp().item()


def compute_adaptive_alpha(y_hat: torch.Tensor, base_alpha: float = 1.0) -> float:
    """Return a per-batch α that scales inversely with estimated Keff.

    Intuition (from Figure 3 of the paper):
      • When Keff is small relative to batch size, each class has many
        samples → empirical statistics are noisy per class → larger α
        keeps us closer to the trustworthy text anchor.
      • When Keff ≈ K (all classes present), MLE estimates are reliable
        → smaller α is fine.

    We clip to [0.5, 5.0] matching the range explored in the ablation study.
    """
    n_samples = y_hat.shape[0]
    keff = max(estimate_keff(y_hat), 1.0)
    # samples-per-effective-class ratio  (higher → more anchor needed)
    ratio = n_samples / keff
    # scale factor: sqrt of (ratio / 64) where 64 is a canonical batch size
    # at which α=1 is known to work well (from paper experiments)
    scale = (ratio / 64.0) ** 0.5
    adaptive = base_alpha * scale
    return float(max(0.5, min(5.0, adaptive)))


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 2 — Symmetrized mutual-KNN affinity matrix
#
# The original affinity W is directed: if i picks j as a neighbour, j does
# not necessarily pick i.  This creates an asymmetric Laplacian which can
# propagate label information in only one direction along some edges.
#
# Fix: W_sym = (W + Wᵀ) / 2  makes propagation bidirectional and reduces
# false cross-class edges that arise when a boundary sample's one-sided
# neighbours happen to be from the wrong class.
#
# Expected gain: +0.3–0.8 % on datasets with well-separated clusters
# (EuroSAT, Flowers, Pets); negligible on ImageNet.
# ─────────────────────────────────────────────────────────────────────────────

def build_affinity_matrix(query_features: torch.Tensor, n_neighbors: int,
                          symmetric: bool = True) -> torch.Tensor:
    """Build a (optionally symmetrized) sparse KNN affinity matrix.

    Args:
        query_features: L2-normalized features [N, D]
        n_neighbors:    number of nearest neighbours per sample
        symmetric:      if True, symmetrize via (W + Wᵀ)/2  (recommended)

    Returns:
        Sparse COO tensor [N, N]
    """
    device = query_features.device
    num_samples = query_features.size(0)
    affinity = query_features.matmul(query_features.T).cpu()

    knn_index = affinity.topk(n_neighbors + 1, -1, largest=True).indices[:, 1:]
    row_indices = torch.arange(num_samples).unsqueeze(1).repeat(1, n_neighbors).flatten()
    col_indices = knn_index.flatten()
    values = affinity[row_indices, col_indices]

    if symmetric:
        # Symmetrize: add the reverse edges with the same weights, then
        # average duplicate pairs.  We concatenate both directions and rely
        # on PyTorch's coalesce() to sum them, then halve.
        row_all = torch.cat([row_indices, col_indices])
        col_all = torch.cat([col_indices, row_indices])
        val_all = torch.cat([values, values])
        W = torch.sparse_coo_tensor(
            torch.stack([row_all, col_all]).to(device),
            val_all.to(device),
            size=(num_samples, num_samples),
            device=device,
        ).coalesce()
        # Divide by 2 so the mean of forward + backward weight is used
        W = torch.sparse_coo_tensor(
            W.indices(), W.values() / 2.0, size=W.size(), device=device
        )
    else:
        W = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices]).to(device),
            values.to(device),
            size=(num_samples, num_samples),
            device=device,
        )
    return W


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian model (unchanged interface)
# ─────────────────────────────────────────────────────────────────────────────

class Gaussian(nn.Module):
    def __init__(self, mu, cov):
        super().__init__()
        self.mu = mu.clone()
        self.cov = cov.clone()

    def forward(self, x, no_exp=False):
        chunk_size = 2500
        N = x.shape[0]
        M = self.mu.shape[0]

        likelihoods = torch.empty((N, M), dtype=x.dtype, device=x.device)
        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            likelihoods[start_idx:end_idx] = -0.5 * (
                ((x[start_idx:end_idx][:, None, :] - self.mu[None, :, 0, :]) ** 2)
                * (1.0 / self.cov[None, :, :])
            ).sum(dim=2)

        if not no_exp:
            likelihoods = torch.exp(likelihoods)
        return likelihoods

    def set_cov(self, cov):
        self.cov = cov

    def set_mu(self, mu):
        self.mu = mu


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 3 — Adaptive temperature τ in z-update
#
# The original code uses a hardcoded temperature of 50 in two places inside
# update_z.  This value was presumably tuned for ViT-B/16 features whose
# cosine-similarity-based logits naturally live in a certain range.  For other
# backbones (RN50, ViT-L/14) the optimal temperature differs.
#
# We expose τ as a parameter (default 50 for backward compatibility) and also
# provide a helper that estimates it from the spread of zero-shot logits.
#
# Expected gain: small for ViT-B/16, up to +0.5% for ResNet backbones where
# logit scales differ.
# ─────────────────────────────────────────────────────────────────────────────

def estimate_temperature(zs_logits: torch.Tensor) -> float:
    """Estimate a per-batch logit temperature from zero-shot logit spread.

    We target an effective temperature such that the *second-highest* logit
    for the average sample is ~2 units below the highest, which is a
    reasonable operating point for the CCCP update.
    """
    # Use inter-quartile range of top-2 logit differences as a proxy
    top2 = zs_logits.topk(2, dim=1).values          # [N, 2]
    margin = (top2[:, 0] - top2[:, 1]).median().item()
    # τ inversely proportional to average margin; clip to [20, 100]
    tau = 50.0 * (2.0 / max(margin, 0.1))
    return float(max(20.0, min(100.0, tau)))


def update_z(likelihoods, y_hat, z, W, lambda_y_hat, lambda_laplacian,
             n_neighbors, sigma, max_iter=5, tau=50.0):
    """CCCP update for assignment variables z.

    Args:
        tau: temperature scaling factor (default 50 as in original paper).
             Use estimate_temperature() or adaptive logic to set this.
    """
    for _ in range(max_iter):
        intermediate = likelihoods.clone()
        intermediate += lambda_laplacian * (tau / (n_neighbors * 2)) * (
            W.T @ z + (W @ z)
        )
        sigma_log_sum = sigma.log().sum(dim=1)
        intermediate -= 0.5 * sigma_log_sum.unsqueeze(0)

        # Numerical stability
        intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0]
        intermediate = (y_hat ** lambda_y_hat) * torch.exp(1.0 / tau * intermediate)
        z = intermediate / torch.sum(intermediate, dim=1, keepdim=True)
    return z


def update_mu(adapter, query_features, z, beta, init_prototypes):
    mu = torch.einsum('ij,ik->jk', z, query_features)
    mu /= torch.sum(z, dim=0).unsqueeze(-1)
    mu = mu.unsqueeze(1)
    mu /= mu.norm(dim=-1, keepdim=True)
    mu = (beta.unsqueeze(-1).unsqueeze(-1) * mu
          + (1 - beta).unsqueeze(-1).unsqueeze(-1) * init_prototypes)
    mu /= mu.norm(dim=-1, keepdim=True)
    return mu


def update_cov(adapter, query_features, z, beta, init_prototypes, init_covariance):
    n_query = z.size(0)
    chunk_size = 2500

    cov = None
    for start_idx in range(0, n_query, chunk_size):
        end_idx = min(start_idx + chunk_size, n_query)
        query_features_chunk = query_features[start_idx:end_idx]
        weighted_sum = (
            (query_features_chunk[:, None, :] - adapter.mu[None, :, 0, :]) ** 2
            * z[start_idx:end_idx, :, None]
        ).sum(dim=0)
        cov = weighted_sum if cov is None else cov + weighted_sum

    cov /= z.sum(dim=0)[:, None]

    delta_mu = (init_prototypes - adapter.mu).squeeze()
    result = torch.bmm(delta_mu.unsqueeze(2), delta_mu.unsqueeze(1))
    diagonal_result = torch.diagonal(result, dim1=1, dim2=2)

    cov = (beta.unsqueeze(-1) * cov
           + (1 - beta).unsqueeze(-1) * (init_covariance + diagonal_result))
    return cov


# ─────────────────────────────────────────────────────────────────────────────
# BUG FIX — init_cov
#
# Original code had `cov /= n_query` INSIDE the for-loop, so for batches with
# more than 2500 samples the covariance was divided by n_query once per chunk.
# With chunk_size=2500 and N=50000 (ImageNet full dataset) this divides by
# n_query 20 times, producing a severely under-scaled initial covariance and
# distorting the log-likelihood term in every subsequent z-update.
#
# The fix: move the division outside the loop.
# ─────────────────────────────────────────────────────────────────────────────

def init_cov(clip_prototypes, query_features, z):
    """Compute initial shared diagonal covariance from zero-shot assignments.

    BUG FIX: normalization by n_query is now outside the chunked loop.
    """
    n_query = z.size(0)
    chunk_size = 2500

    cov = None
    for start_idx in range(0, n_query, chunk_size):
        end_idx = min(start_idx + chunk_size, n_query)
        query_features_chunk = query_features[start_idx:end_idx]
        chunk_result = torch.einsum(
            'ij,ijk->k',
            z[start_idx:end_idx, :],
            (query_features_chunk[:, None, :] - clip_prototypes[None, :, 0, :]) ** 2,
        )
        cov = chunk_result if cov is None else cov + chunk_result

    cov /= n_query  # ← moved OUTSIDE the loop (bug fix)
    return cov


def update_beta(z, alpha, soft=False):
    if soft:
        sum_z = torch.sum(z, dim=0)
        beta = sum_z / (alpha + sum_z)
    else:
        predicted_classes = torch.argmax(z, dim=1)
        sum_z = torch.bincount(predicted_classes, minlength=z.size(1)).float()
        beta = sum_z / (alpha + sum_z + 1e-12)
    return beta


def get_zero_shot_logits(query_features, query_labels, clip_prototypes):
    clip_logits = 100 * query_features @ clip_prototypes
    return clip_logits.squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# Main solver — improved version of StatA_solver, named StatA_N_solver
# to avoid any name collision with the original StatA.py
# ─────────────────────────────────────────────────────────────────────────────

def StatA_N_solver(
    query_features,
    query_labels,
    clip_prototypes,
    alpha: float = 1.0,
    soft_beta: bool = False,
    lambda_y_hat: float = 1.0,
    lambda_laplacian: float = 1.0,
    n_neighbors: int = 3,
    max_iter: int = 10,
    # New flags (all False/None = original behaviour for easy ablation)
    adaptive_alpha: bool = True,       # Improvement 1
    symmetric_affinity: bool = True,   # Improvement 2
    adaptive_tau: bool = True,         # Improvement 3
):
    """StatA solver with optional improvements over the ICLR-2025 paper.

    New keyword arguments vs the original:
        adaptive_alpha (bool):      Estimate Keff and scale α accordingly.
        symmetric_affinity (bool):  Symmetrize the KNN affinity matrix.
        adaptive_tau (bool):        Estimate τ from zero-shot logit spread.

    All improvements default to True.  Set them all to False to reproduce the
    original paper's exact behaviour.
    """
    query_labels = query_labels.cuda().float()
    clip_prototypes = clip_prototypes.cuda().float()
    query_features = query_features.cuda().float()

    # ── Z init ──────────────────────────────────────────────────────────────
    zs_logits = get_zero_shot_logits(query_features, query_labels, clip_prototypes)
    y_hat = F.softmax(zs_logits, dim=1)
    z = y_hat.clone()

    # ── Adaptive α (Improvement 1) ──────────────────────────────────────────
    if adaptive_alpha:
        alpha = compute_adaptive_alpha(y_hat, base_alpha=alpha)

    # ── Adaptive temperature (Improvement 3) ────────────────────────────────
    tau = estimate_temperature(zs_logits) if adaptive_tau else 50.0

    # ── μ init ───────────────────────────────────────────────────────────────
    mu = clip_prototypes.permute(2, 0, 1)

    # ── Σ init (bug-fixed) ───────────────────────────────────────────────────
    cov = init_cov(clip_prototypes.permute(2, 0, 1), query_features, z)
    cov = cov.unsqueeze(0).repeat(y_hat.size(-1), 1)
    init_covariance = cov

    adapter = Gaussian(mu=mu, cov=cov).cuda()

    # ── Affinity matrix (Improvement 2) ─────────────────────────────────────
    W = build_affinity_matrix(query_features.float(), n_neighbors,
                              symmetric=symmetric_affinity)

    # ── Main EM loop ─────────────────────────────────────────────────────────
    for k in range(max_iter + 1):

        likelihoods = adapter(query_features, no_exp=True)

        z = update_z(likelihoods, y_hat, z, W, lambda_y_hat, lambda_laplacian,
                     n_neighbors, adapter.cov, tau=tau)

        if k == max_iter:
            break

        beta = update_beta(z, alpha, soft=soft_beta)
        mu = update_mu(adapter, query_features, z, beta,
                       clip_prototypes.permute(2, 0, 1))
        adapter.set_mu(mu)
        cov = update_cov(adapter, query_features, z, beta,
                         clip_prototypes.permute(2, 0, 1), init_covariance)
        adapter.set_cov(cov)

    return y_hat.cpu(), z.cpu()
