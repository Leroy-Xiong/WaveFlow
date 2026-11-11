import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import autocast


class FlowDiffusion:
    def __init__(self, model, device='cuda', prediction_type='v_pred', t_eps=1e-6, use_amp=False, grad_clip_norm=None, channel_weights=None,
                 extreme_loss_weight=0.0, extreme_loss_thresh=2.5, extreme_asym=0.0,
                 extreme_weight_channels=None, t_noise_df=None, t_noise_gate=None,
                 objective='cfm', mc_dropout=False):
        """
        Standard Conditional Flow Matching Framework.

        Args:
            model: Neural network v_theta(t, x, **condition_kwargs) that predicts velocity or x.
            device: 'cuda' or 'cpu'.
            prediction_type: Type of prediction - 'v_pred' (predict velocity, v-loss) or 
                            'x_pred' (predict x, v-loss). Default is 'v_pred'.
            t_eps: Small epsilon to avoid division by zero when t is close to 1.
            use_amp: Whether to use automatic mixed precision (AMP) with BF16 for faster training.
            grad_clip_norm: Max norm for gradient clipping. None or 0 means no clipping.
            channel_weights: Optional per-channel loss weights (list of floats, length C).
                            Applied to the squared error before averaging. None = uniform.
            extreme_loss_weight: Extra weight on extreme hs samples (normalized space):
                            se *= 1 + extreme_loss_weight * relu(hs_ch - extreme_loss_thresh).
                            0.0 = uniform weighting (default).
            extreme_loss_thresh: hs channel threshold (normalized) above which the
                            extreme up-weighting kicks in (default 2.5 ~ physical 2m).
            extreme_asym: Asymmetric (Exloss-style) multiplier for extreme hs samples.
                            Scalar float: applies to channel 0 (hs) only (legacy).
                            List of floats (len = in_channels): per-channel asymmetric
                            factor; channel c underestimates of the velocity get an
                            extra factor (1 + asym[c] * frac_under[c]). Penalizes
                            under-prediction more than over-prediction. 0.0 = symmetric.
            extreme_weight_channels: Channel indices (0-based) to which the base
                            extreme gate weight applies. None = [0] (legacy: hs only).
                            E.g. [0,1,2,3,4] applies the storm gate to all channels.
            t_noise_df: Degrees of freedom nu for a Student-t prior (t-Flow,
                            arXiv:2410.14171) replacing the Gaussian prior:
                            n = eps / sqrt(kappa), eps ~ N(0,I), kappa ~ Gamma(nu/2, nu/2).
                            None or inf = standard Gaussian (legacy behavior).
                            Scalar float = one shared kappa per sample (multivariate t,
                            paper Eq. 164). List of len(in_channels) = per-channel df
                            (nu=inf entries stay Gaussian). Training and sampling both
                            draw from this prior, so they stay consistent. nu must be > 0;
                            variance of the prior is nu/(nu-2) for nu > 2 (paper does NOT
                            rescale to unit variance).
            t_noise_gate: Optional dict {'s_lo': float, 's_hi': float} enabling
                            condition-gated heavy tails: the Student-t kappa is
                            interpolated with 1 (Gaussian) by a per-sample storm gate
                            g in [0,1] computed from the ERA5 condition's max 10m wind
                            magnitude s: g = clamp((s - s_lo)/(s_hi - s_lo)).
                            kappa_eff = 1 + g * (kappa - 1): g=0 -> exact Gaussian
                            (calm), g=1 -> full Student-t (storm). The gate uses ONLY
                            the condition, so train/sample stay consistent. Requires
                            t_noise_df to be set.
        """
        self.model = model.to(device)
        self.device = device
        self.prediction_type = prediction_type
        self.t_eps = t_eps
        self.use_amp = use_amp
        # Training objective / sampling mode (UQ method comparison):
        #   'cfm'  - conditional flow matching (default, the paper's method)
        #   'mse'  - deterministic regression: model(zeros, t=1, cond) -> x1,
        #            MSE loss (deep-ensemble members; MC dropout when model has
        #            dropout layers and sampling.mc_dropout=true)
        #   'nll'  - heteroscedastic Gaussian regression: model outputs
        #            [mu, log_sigma] (out_channels = 2*in_channels), Gaussian NLL
        if objective not in ('cfm', 'mse', 'nll'):
            raise ValueError(f"Invalid objective '{objective}'. Must be 'cfm', 'mse' or 'nll'.")
        self.objective = objective
        # MC-dropout sampling: keep nn.Dropout / SDPA dropout active at sample
        # time so each member is a different dropout realization
        self.mc_dropout = bool(mc_dropout)
        self.grad_clip_norm = grad_clip_norm if grad_clip_norm and grad_clip_norm > 0 else None
        self.channel_weights = (
            torch.tensor(channel_weights, dtype=torch.float32, device=device)
            if channel_weights is not None else None
        )
        self.extreme_loss_weight = float(extreme_loss_weight if extreme_loss_weight else 0.0)
        self.extreme_loss_thresh = float(extreme_loss_thresh)
        if isinstance(extreme_asym, (list, tuple)):
            self.extreme_asym = [float(a) if a else 0.0 for a in extreme_asym]
        else:
            self.extreme_asym = float(extreme_asym if extreme_asym else 0.0)
        self.extreme_weight_channels = (
            [int(c) for c in extreme_weight_channels]
            if extreme_weight_channels is not None else [0])
        if t_noise_df is None:
            self.t_noise_df = None
        elif isinstance(t_noise_df, (list, tuple)):
            self.t_noise_df = [float(nu) for nu in t_noise_df]
        else:
            self.t_noise_df = float(t_noise_df)
        # Condition-gated heavy tail (see t_noise_gate docstring above)
        if t_noise_gate is not None:
            if self.t_noise_df is None:
                raise ValueError("t_noise_gate requires t_noise_df to be set")
            lo = float(t_noise_gate['s_lo'])
            hi = float(t_noise_gate['s_hi'])
            if not hi > lo:
                raise ValueError(f"t_noise_gate requires s_hi > s_lo, got {lo}, {hi}")
            self.t_noise_gate_lo, self.t_noise_gate_hi = lo, hi
        else:
            self.t_noise_gate_lo = None
        self._validate_t_noise_df()
        
        # NOTE: no GradScaler — AMP here is BF16, which has FP32-range exponents
        # and does not need loss scaling. A scaler only adds per-step GPU syncs
        # (unscale_ / found_inf checks) with zero numerical benefit for BF16.
        
        if prediction_type not in ['v_pred', 'x_pred']:
            raise ValueError(f"Invalid prediction_type: {prediction_type}. Must be 'v_pred' or 'x_pred'.")

    def _validate_t_noise_df(self):
        """Validate t_noise_df: every finite nu must be > 0 (nu > 2 for finite variance)."""
        if self.t_noise_df is None:
            return
        nus = self.t_noise_df if isinstance(self.t_noise_df, list) else [self.t_noise_df]
        for nu in nus:
            if math.isinf(nu):
                continue
            if nu <= 0.0:
                raise ValueError(f"t_noise_df entries must be > 0 (or inf), got {nu}")

    def _sample_prior_noise(self, shape, dtype=torch.float32, era5_data=None):
        """
        Draw prior noise of the given shape on self.device.
        t_noise_df None: standard Gaussian N(0, I) (legacy).
        Otherwise Student-t (t-Flow): n = eps / sqrt(kappa), eps ~ N(0,I),
        kappa ~ Gamma(nu/2, nu/2) (E[kappa] = 1; heavy tails come from small kappa).
        Scalar nu -> one kappa per sample, shared across all dims (multivariate t).
        List nu  -> one kappa per (sample, channel); nu=inf channels stay Gaussian.
        If t_noise_gate is configured, kappa is smoothly interpolated towards 1
        (Gaussian) by the per-sample storm gate from the ERA5 condition:
        kappa_eff = 1 + g * (kappa - 1). era5_data must be provided then.
        """
        eps = torch.randn(shape, device=self.device, dtype=dtype)
        if self.t_noise_df is None:
            return eps
        if isinstance(self.t_noise_df, list):
            C = shape[-1]
            if len(self.t_noise_df) != C:
                raise ValueError(
                    f"t_noise_df list length {len(self.t_noise_df)} != channels {C}")
            kappa_shape = [shape[0]] + [1] * (len(shape) - 2) + [C]
            kappa = torch.ones(kappa_shape, device=self.device, dtype=torch.float32)
            for c, nu in enumerate(self.t_noise_df):
                if math.isinf(nu):
                    continue
                kappa[..., c] = self._sample_kappa(nu, kappa_shape[:-1])
            gate_view = [shape[0]] + [1] * (len(shape) - 1)
        else:
            nu = self.t_noise_df
            if math.isinf(nu):
                return eps
            kappa = self._sample_kappa(nu, [shape[0]] + [1] * (len(shape) - 1))
            gate_view = [shape[0]] + [1] * (len(shape) - 1)
        if self.t_noise_gate_lo is not None:
            if era5_data is not None:
                g = self._storm_gate(era5_data).view(gate_view)
            else:
                # No condition available (unconditional sampling): keep full tail.
                g = torch.ones(gate_view, device=self.device, dtype=torch.float32)
            kappa = 1.0 + g * (kappa - 1.0)
        return eps * torch.rsqrt(kappa).to(dtype)

    def _storm_gate(self, era5_data):
        """
        Per-sample storm gate g in [0,1] from the normalized ERA5 condition:
        s = max over (T, H, W) of the 10m wind magnitude; g = clamp((s - s_lo)/(s_hi - s_lo)).
        Uses ONLY condition data -> identical at train and sample time (same quantity as
        compute_extreme_feats' wind_max, cf. patch_transformer.py).
        """
        if era5_data.ndim == 4:
            era5_data = era5_data.unsqueeze(1)  # (B, 1, 3, H, W)
        u = era5_data[:, :, 0].float()
        v = era5_data[:, :, 1].float()
        s = torch.sqrt(u * u + v * v).amax(dim=(1, 2, 3))  # (B,)
        return ((s - self.t_noise_gate_lo) / (self.t_noise_gate_hi - self.t_noise_gate_lo)).clamp(0.0, 1.0)

    def _sample_kappa(self, nu, sample_shape):
        """kappa ~ Gamma(nu/2, nu/2) (i.e. chi2(nu)/nu), on self.device."""
        half = nu / 2.0
        dist = torch.distributions.Gamma(
            torch.tensor(half, device=self.device),
            torch.tensor(half, device=self.device))
        return dist.sample(sample_shape)

    def compute_loss(self, x1, mask=None, **condition_kwargs):
        """
        Computes the Conditional Flow Matching (CFM) loss with condition.

        Args:
            x1: Real data batch of shape [batch_size, *dims].
            mask: Optional boolean tensor same shape as x1 (True = valid).
                  When provided, loss is computed only over valid positions.
            **condition_kwargs: Flexible keyword arguments for different model
                               conditions (era5_data, etc.).
        """
        batch_size = x1.shape[0]

        if self.objective in ('mse', 'nll'):
            # Deterministic / heteroscedastic regression baselines (UQ comparison):
            # the model reads a zero state at t=1 and predicts x1 directly.
            t_one = torch.ones(batch_size, device=self.device)
            x_zero = torch.zeros_like(x1)
            out = self.model(x_zero, t_one, **condition_kwargs)
            if self.objective == 'mse':
                pred_v, target_v = out, x1
                se = (pred_v - target_v) ** 2
            else:  # 'nll': out = [mu, log_sigma] over 2*in_channels
                mu, log_sigma = out.chunk(2, dim=-1)
                log_sigma = log_sigma.clamp(-6.0, 4.0)
                pred_v, target_v = mu, x1
                # Gaussian NLL elementwise (up to additive const): shared weighting below
                se = 0.5 * ((target_v - mu) / log_sigma.exp()) ** 2 + log_sigma
        else:
            # CFM objective (default)
            # 1. Sample time t uniformly from [0, 1]
            t_shape = [batch_size] + [1] * (x1.dim() - 1)
            t = torch.rand(t_shape, device=self.device)

            # 2. Sample source noise x0 from the prior (Gaussian or Student-t if t_noise_df set;
            #    condition-gated tail if t_noise_gate set)
            x0 = self._sample_prior_noise(x1.shape, dtype=x1.dtype,
                                          era5_data=condition_kwargs.get('era5_data'))

            # 3. Compute the interpolation (OT path) at time t
            xt = (1 - t) * x0 + t * x1

            if self.prediction_type == 'v_pred':
                # v-pred: Model predicts velocity directly
                target_v = x1 - x0
                pred_v = self.model(xt, t.squeeze(), **condition_kwargs)
            elif self.prediction_type == 'x_pred':
                # x-pred: Model predicts x, convert to velocity (JiT style)
                target_v = (x1 - xt) / (1 - t).clamp_min(self.t_eps)
                pred_x = self.model(xt, t.squeeze(), **condition_kwargs)
                pred_v = (pred_x - xt) / (1 - t).clamp_min(self.t_eps)

            # 5. Compute MSE loss on velocity (masked if requested)
            se = (pred_v - target_v) ** 2
        if self.channel_weights is not None:
            # Broadcast per-channel weights over the last (channel) dim
            se = se * self.channel_weights
        if self.extreme_loss_weight > 0.0:
            # Up-weight flow-matching fit on extreme sea states. x1[..., 0] is the
            # normalized hs channel (log1p+zscore; physical 2m ~ z=2.65, 4m ~ z=4.5).
            # The storm gate (by hs magnitude) can apply to any set of channels
            # (extreme_weight_channels); the asymmetric under-prediction penalty
            # (extreme_asym) can be scalar (hs only) or per-channel.
            # NOTE: no in-place indexing — the asymmetric weight depends on pred_v
            # (gradient path), so `se[..., 0] = ...` would trip autograd version checks.
            hs_x1 = x1[..., 0]
            gate = torch.relu(hs_x1 - self.extreme_loss_thresh)
            w = 1.0 + self.extreme_loss_weight * gate  # (...,)
            asym = self.extreme_asym
            asym_is_list = isinstance(asym, list)
            if (asym_is_list and any(a > 0.0 for a in asym)) or \
                    (not asym_is_list and asym > 0.0):
                # Exloss-style asymmetric weighting: velocity underestimates of
                # extreme hs (pred_v < target_v -> sample lands too low) get an
                # extra factor, penalizing under-prediction more than over.
                e = pred_v - target_v
                frac_under = torch.clamp(-e, min=0.0) / (e.abs().clamp_min(1e-6))
                w_full = []
                for c in range(se.shape[-1]):
                    wc = w if c in self.extreme_weight_channels else torch.ones_like(w)
                    ac = asym[c] if asym_is_list else (asym if c == 0 else 0.0)
                    if ac > 0.0:
                        wc = wc * (1.0 + ac * frac_under[..., c])
                    w_full.append(wc)
                se = se * torch.stack(w_full, dim=-1)
            else:
                w_full = [w if c in self.extreme_weight_channels else torch.ones_like(w)
                          for c in range(se.shape[-1])]
                se = se * torch.stack(w_full, dim=-1)
        if mask is not None:
            loss = se[mask].mean()
        else:
            loss = se.mean()
        return loss

    def train_step(self, x1, optimizer, debug_grad=False, **condition_kwargs):
        """
        Performs a single training step with condition.
        
        Args:
            x1: Real data batch.
            optimizer: Optimizer instance.
            debug_grad: If True, return per-layer gradient statistics for debugging.
            **condition_kwargs: Flexible condition inputs for the model.
                              e.g., era5_data=..., condition=..., context=..., labels=...
        
        Returns:
            tuple: (loss, grad_norm, grad_debug_info or None)
                   loss and grad_norm are detached 0-dim GPU tensors in the normal
                   path (the caller decides when to sync via .item()/float());
                   they are float('nan') when the step is skipped.
        
        Performance notes (do not regress):
          - Exactly ONE host<->device sync per step: the combined finite check on
            (loss, grad_norm). Per-parameter NaN/Inf scans were removed — they
            caused hundreds of blocking syncs per step (~36% of step CPU time).
          - A non-finite total grad norm implies a NaN/Inf gradient exists, so
            skipping the step on that condition is equivalent to the old scan.
        """
        optimizer.zero_grad()

        # Use automatic mixed precision (BF16) if enabled
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            loss = self.compute_loss(x1, **condition_kwargs)
        
        loss.backward()
        
        # Debug: collect per-layer gradient stats (only when explicitly requested)
        grad_debug_info = None
        if debug_grad:
            grad_debug_info = self._get_gradient_debug_info()
        
        # Total grad norm via foreach ops (no per-parameter syncs)
        if self.grad_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_norm
            )
        else:
            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            grad_norm = torch.linalg.vector_norm(
                torch.stack(torch._foreach_norm(grads))
            )
        
        # Single sync per step: non-finite loss or grad norm -> skip the step.
        # (If the loss were NaN/Inf, backward propagates it into the grads and
        # the total norm becomes non-finite as well, so this one check covers both.)
        if not torch.isfinite(grad_norm).item() or not torch.isfinite(loss.detach()).item():
            print(f"[WARNING] Non-finite loss/grad-norm detected "
                  f"(loss={float(loss.detach())}, grad_norm={float(grad_norm)}); skipping step")
            optimizer.zero_grad()
            return float('nan'), float('nan'), grad_debug_info
        
        optimizer.step()
        
        return loss.detach(), grad_norm, grad_debug_info
        
    def _get_gradient_debug_info(self):
        """
        Collect per-layer gradient statistics for debugging gradient explosion.
        Returns a dict with layer names and their gradient norms.
        """
        debug_info = {}
        # Get the underlying model (unwrap DDP if necessary)
        model = self.model.module if hasattr(self.model, 'module') else self.model
            
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.data.norm(2).item()
                grad_max = p.grad.data.abs().max().item()
                grad_min = p.grad.data.min().item()
                    
                # Only log layers with unusually large gradients
                if grad_norm > 10.0 or grad_max > 100.0:
                    debug_info[name] = {
                        'norm': grad_norm,
                        'max': grad_max,
                        'min': grad_min,
                        'shape': list(p.shape)
                    }
            
        return debug_info

    @torch.no_grad()
    def sample(self, batch_size, data_shape, steps=100, save_times=None,
               stochastic_gamma=0.0, stochastic_schedule="const", prior_noise=None,
               **condition_kwargs):
        """
        Generates samples based on the provided condition.
        
        Args:
            batch_size: Number of samples to generate.
            data_shape: Shape of a SINGLE data point (e.g., (channels, H, W) or (dim,)).
                        Output shape will be (batch_size, *data_shape).
            steps: Number of integration steps.
            save_times: Optional list of ODE times t in [0, 1] at which to snapshot
                        the intermediate state x_t (e.g., [0, 0.25, 0.5]).
                        t is the mathematical ODE time: t=0 is pure noise, t=1 is
                        the final sample. Snapshot step indices are rounded to the
                        nearest Euler step. None = disabled (default, old behavior).
            stochastic_gamma: Noise injection amplitude for stochastic (SDE-style)
                        sampling: x += v*dt + sigma(t)*sqrt(dt)*z. 0.0 (default)
                        = deterministic Euler ODE (legacy behavior).
            stochastic_schedule: sigma(t) shape when stochastic_gamma > 0:
                        'const'  -> sigma(t) = gamma (uniform noise at all t)
                        'linear' -> sigma(t) = gamma * t (more noise near t=1)
            **condition_kwargs: Flexible condition inputs for the model.
                              e.g., era5_data=..., condition=..., context=..., labels=...
        
        Returns:
            If save_times is None: final sample tensor x.
            Otherwise: tuple (x, snapshots) where snapshots is a dict
                       {t_value: tensor of shape (batch_size, *data_shape)}.
        """
        self.model.eval()

        if self.objective in ('mse', 'nll'):
            # Regression baselines: one forward pass per sample (no ODE).
            if self.mc_dropout:
                # MC-dropout sampling: keep every dropout-capable module in train
                # mode (timm Mlp uses nn.Dropout; StableAttention/CrossAttention
                # gate SDPA dropout_p on their own .training flag).
                for m in self.model.modules():
                    if isinstance(m, nn.Dropout) or getattr(m, 'attn_drop', 0.0):
                        m.train()
            if save_times:
                raise ValueError("save_times (ODE snapshots) is not supported for "
                                 f"objective='{self.objective}'")
            t_one = torch.ones(batch_size, device=self.device)
            x_zero = torch.zeros(batch_size, *data_shape, device=self.device)
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                out = self.model(x_zero, t_one, **condition_kwargs)
            out = out.float()
            if self.objective == 'mse':
                return out
            # 'nll': sample a member from the predicted Gaussian N(mu, sigma)
            mu, log_sigma = out.chunk(2, dim=-1)
            log_sigma = log_sigma.clamp(-6.0, 4.0)
            return mu + log_sigma.exp() * torch.randn_like(mu)

        # 1. Initialize from Prior Noise x(0) (Gaussian, or Student-t if t_noise_df set;
        #    condition-gated tail if t_noise_gate set). prior_noise overrides the draw
        #    (input-perturbation UQ baseline shares one noise across ensemble members).
        full_shape = (batch_size, *data_shape)
        if prior_noise is not None:
            if tuple(prior_noise.shape) != full_shape:
                raise ValueError(
                    f"prior_noise shape {tuple(prior_noise.shape)} != required {full_shape}")
            x = prior_noise.to(device=self.device, dtype=torch.float32)
        else:
            x = self._sample_prior_noise(full_shape,
                                         era5_data=condition_kwargs.get('era5_data'))
        
        dt = 1.0 / steps
        stochastic_gamma = float(stochastic_gamma if stochastic_gamma else 0.0)
        
        # Resolve requested snapshot times to step indices (step i holds x at t=i/steps)
        snapshots = {}
        snapshot_steps = {}
        if save_times:
            for t_req in save_times:
                t_req = float(t_req)
                if not 0.0 <= t_req <= 1.0:
                    raise ValueError(f"save_times values must be in [0, 1], got {t_req}")
                step_idx = min(int(round(t_req * steps)), steps)
                snapshot_steps[step_idx] = t_req
        
        # 2. ODE Integration Loop (Euler Method, optionally stochastic)
        for i in range(steps):
            # Snapshot the state at t=i/steps if requested (covers t=0 initial noise)
            if i in snapshot_steps:
                snapshots[snapshot_steps[i]] = x.clone()
            
            # Current time t
            t_val = i / steps
            # Create t vector [batch_size]
            t = torch.full((batch_size,), t_val, device=self.device)
            
            # Use automatic mixed precision (BF16) if enabled
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                if self.prediction_type == 'v_pred':
                    # v-pred: Model predicts velocity directly
                    v = self.model(x, t, **condition_kwargs)
                elif self.prediction_type == 'x_pred':
                    # x-pred: Model predicts x, convert to velocity (JiT style)
                    # v = (x_pred - x_t) / (1 - t)
                    pred_x = self.model(x, t, **condition_kwargs)
                    # Use clamp_min on denominator to avoid division by zero (matching JiT implementation)
                    t_expanded = t.view(batch_size, *([1] * (x.dim() - 1)))
                    v = (pred_x - x) / (1 - t_expanded).clamp_min(self.t_eps)
            
            # Update state: x(t+dt) = x(t) + v * dt
            if stochastic_gamma > 0.0:
                if stochastic_schedule == "linear":
                    sigma_t = stochastic_gamma * t_val
                else:
                    sigma_t = stochastic_gamma
                x = x + v * dt + sigma_t * math.sqrt(dt) * torch.randn_like(x)
            else:
                x = x + v * dt
        
        # Snapshot the final state at t=1 if requested
        if steps in snapshot_steps:
            snapshots[snapshot_steps[steps]] = x.clone()
        
        if save_times:
            return x, snapshots
        return x