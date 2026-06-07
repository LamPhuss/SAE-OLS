"""
LLM Forward Hook Utilities for SAE-OLS.

Provides mechanisms to:
1. Intercept hidden states at a specific transformer layer
2. Modify hidden states in-place during the forward pass (for watermark steering)

This is the white-box access mechanism — we register PyTorch hooks on the
target residual stream layer to read and modify hidden representations.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable


class HiddenStateInterceptor:
    """
    Intercepts and optionally modifies hidden states at a specific transformer layer.

    Usage during generation:
        interceptor = HiddenStateInterceptor(model, layer_idx=20)
        interceptor.register()

        # Set a modifier function that will be called on each forward pass
        interceptor.set_modifier(my_steering_function)

        # Generate text — hidden states are modified in-place
        output = model.generate(...)

        interceptor.remove()
    """

    def __init__(self, model: nn.Module, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self._handle: Optional[torch.utils.hooks.RemovableHook] = None
        self._captured_hidden_state: Optional[torch.Tensor] = None
        self._modifier: Optional[Callable] = None

    def _hook_fn(self, module, inputs, outputs):
        """
        Hook function registered on the target layer.
        Captures the hidden state and optionally applies a modifier.
        """
        # outputs can be a tuple (hidden_states, ...) or a BaseModelOutputWithPast
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        # Ensure 3D: [batch, seq_len, d_model]
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        self._captured_hidden_state = hidden_states.detach().clone()

        if self._modifier is not None:
            modified = self._modifier(hidden_states)
            # Return modified outputs tuple
            return (modified,) + outputs[1:]

        return outputs

    def _get_target_layer(self) -> nn.Module:
        """Get the target transformer layer module."""
        # Support common model architectures
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Gemma, Llama, Mistral style
            return self.model.model.layers[self.layer_idx]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2 style
            return self.model.transformer.h[self.layer_idx]
        else:
            raise ValueError(
                f"Unsupported model architecture. Cannot find layer {self.layer_idx}. "
                "Please extend _get_target_layer() for your model."
            )

    def register(self):
        """Register the forward hook on the target layer."""
        layer = self._get_target_layer()
        self._handle = layer.register_forward_hook(self._hook_fn)

    def remove(self):
        """Remove the hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def set_modifier(self, modifier: Optional[Callable]):
        """
        Set a function that modifies hidden states during forward pass.

        The modifier takes hidden_states [batch, seq_len, d_model] and returns
        modified hidden_states of the same shape.
        """
        self._modifier = modifier

    def clear_modifier(self):
        """Remove the modifier (stop steering)."""
        self._modifier = None

    @property
    def captured(self) -> Optional[torch.Tensor]:
        """Get the most recently captured hidden state (before modification)."""
        return self._captured_hidden_state

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()


def gather_residual_activations(
    model: nn.Module,
    target_layer: int,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Simple one-shot extraction of hidden states at a specific layer.
    Used during detection (read-only, no modification).

    Args:
        model: The LLM
        target_layer: Which transformer layer to intercept
        input_ids: Input token IDs, shape [batch, seq_len]
        attention_mask: Optional attention mask

    Returns:
        Hidden states at the target layer, shape [batch, seq_len, d_model]
    """
    captured = None

    def hook_fn(module, inputs, outputs):
        nonlocal captured
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        captured = hidden_states.detach()
        return outputs

    # Get the layer
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layer = model.model.layers[target_layer]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layer = model.transformer.h[target_layer]
    else:
        raise ValueError(f"Unsupported model architecture for layer {target_layer}")

    handle = layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)
    handle.remove()

    return captured
