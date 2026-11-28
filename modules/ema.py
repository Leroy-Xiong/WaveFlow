"""Exponential Moving Average (EMA) for model weights."""

import torch


class EMA:
    """
    Exponential Moving Average (EMA) for model weights.
    
    EMA maintains a shadow copy of model parameters that is updated as:
        shadow = decay * shadow + (1 - decay) * current
    
    This provides a more stable version of the model for inference/sampling.
    """
    
    def __init__(self, model, decay=0.9999, update_after_step=100, device=None):
        """
        Initialize EMA.
        
        Args:
            model: The model to track (can be DDP wrapped).
            decay: EMA decay rate (higher = slower update). Typical: 0.999-0.9999.
            update_after_step: Start EMA updates after this many steps (warmup).
            device: Device to store shadow parameters. If None, uses model's device.
        """
        self.decay = decay
        self.update_after_step = update_after_step
        self.step = 0
        
        # Get the underlying model (unwrap DDP if necessary)
        self.model = model.module if hasattr(model, 'module') else model
        
        # Create shadow parameters as a deep copy
        self.shadow = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
                if device is not None:
                    self.shadow[name] = self.shadow[name].to(device)
        
        # Lazily-built flat (param, shadow) lists for fused foreach updates.
        # Invalidated whenever self.shadow is replaced (load_state_dict).
        self._flat_params = None
        self._flat_shadows = None
    
    @torch.no_grad()
    def update(self):
        """
        Update EMA shadow parameters.
        Should be called after each optimizer step.
        
        Uses torch._foreach_* fused ops: a single kernel launch for all
        parameters instead of one lerp_/copy_ launch per parameter
        (hundreds of launches per step for large models).
        """
        self.step += 1
        
        if self._flat_params is None:
            params, shadows = [], []
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    params.append(param.data)
                    shadows.append(self.shadow[name])
            self._flat_params = params
            self._flat_shadows = shadows
        
        # Skip updates during warmup period: copy current parameters
        if self.step <= self.update_after_step:
            torch._foreach_copy_(self._flat_shadows, self._flat_params)
            return
        
        # Apply EMA update: shadow = lerp(shadow, param, 1 - decay)
        torch._foreach_lerp_(self._flat_shadows, self._flat_params, 1.0 - self.decay)
    
    def state_dict(self):
        """
        Return EMA state for checkpointing.
        """
        return {
            'shadow': self.shadow,
            'decay': self.decay,
            'step': self.step,
            'update_after_step': self.update_after_step
        }
    
    def load_state_dict(self, state_dict):
        """
        Load EMA state from checkpoint.
        """
        self.shadow = state_dict['shadow']
        self.decay = state_dict.get('decay', self.decay)
        self.step = state_dict.get('step', 0)
        self.update_after_step = state_dict.get('update_after_step', self.update_after_step)
        # Shadow tensors were replaced: rebuild the flat foreach cache
        self._flat_params = None
        self._flat_shadows = None
        
        # Move shadow parameters to the same device as model parameters
        # This ensures device consistency after resuming from checkpoint
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = self.shadow[name].to(param.device)
    
    @torch.no_grad()
    def apply_shadow(self):
        """
        Apply EMA shadow parameters to the model.
        Returns the original parameters so they can be restored.
        """
        backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup
    
    @torch.no_grad()
    def restore(self, backup):
        """
        Restore original parameters from backup.
        
        Args:
            backup: Dictionary returned by apply_shadow().
        """
        for name, param in self.model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
