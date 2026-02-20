"""
nano_rust_utils — Universal exporter for PyTorch → NANO-RUST-AI.

Walks a PyTorch model's children, quantizes weights to i8,
and generates both binary files (.bin) and Rust source code (.rs).

Usage:
    from nano_rust_utils import quantize_weights, export_to_rust, export_weights_bin

    # Quantize
    q_weights = quantize_weights(model)

    # Export to Rust code
    rust_code = export_to_rust(model, "my_model")

    # Export binary weights
    export_weights_bin(q_weights, "output/")
"""

import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


def quantize_to_i8(tensor: np.ndarray, scale: float = 127.0) -> Tuple[np.ndarray, float]:
    """Quantize float32 tensor to i8 using symmetric linear scaling.

    Maps [-max_abs, +max_abs] → [-127, +127].

    Returns:
        (quantized_i8_array, scale_factor)
        scale_factor: max_abs / 127. To dequantize: float_value = i8_value * scale_factor
    """
    max_abs = np.max(np.abs(tensor))
    if max_abs == 0:
        return np.zeros_like(tensor, dtype=np.int8), 1.0
    scale_factor = max_abs / scale
    quantized = np.clip(np.round(tensor / scale_factor), -128, 127).astype(np.int8)
    return quantized, scale_factor


def compute_requant_params(
    input_scale: float,
    weight_scale: float,
    output_scale: float,
    max_shift: int = 31,
) -> Tuple[int, int]:
    """Compute TFLite-style fixed-point requantization multiplier and shift.

    The requantization formula is:
        output_i8 = (acc_i32 * M) >> shift

    Where M and shift are chosen so that:
        M / 2^shift ≈ input_scale * weight_scale / output_scale

    This ensures the quantized output has the correct scale relative
    to the actual float32 computation.

    Args:
        input_scale: Scale factor of the input activation (from quantize_to_i8)
        weight_scale: Scale factor of the weights (from quantize_to_i8)
        output_scale: Scale factor of the output (from running float model)
        max_shift: Maximum allowed shift value

    Returns:
        (requant_m, requant_shift): int32 multiplier and uint32 shift
    """
    # The real ratio we want to approximate with integer arithmetic
    real_multiplier = input_scale * weight_scale / output_scale

    # Find shift such that M = real_multiplier * 2^shift fits in int32
    # We want M in [2^30, 2^31) for maximum precision
    shift = 0
    while shift < max_shift:
        m = real_multiplier * (1 << shift)
        if m >= (1 << 30):
            break
        shift += 1

    requant_m = int(round(real_multiplier * (1 << shift)))
    # Clamp to avoid overflow
    requant_m = min(requant_m, (1 << 31) - 1)

    return requant_m, shift

def compute_activation_scale_params(
    input_scale: float,
    lut_divisor: float,
    max_shift: int = 15,
) -> Tuple[int, int]:
    """Compute fixed-point (mult, shift) for activation LUT rescaling.

    The LUT assumes input = i8_val / lut_divisor (16 for sigmoid, 32 for tanh).
    But actual float value = i8_val * input_scale.
    We need: rescaled_index = i8_val * input_scale * lut_divisor
    So: scale_mult / 2^scale_shift ≈ input_scale * lut_divisor
    """
    real_factor = input_scale * lut_divisor
    shift = 0
    while shift < max_shift:
        m = real_factor * (1 << shift)
        if m >= (1 << 14):  # Keep smaller to avoid overflow on i8*i32
            break
        shift += 1
    scale_mult = int(round(real_factor * (1 << shift)))
    scale_mult = min(scale_mult, (1 << 15) - 1)  # i16-safe range
    return scale_mult, shift


def calibrate_model(
    model,
    input_tensor,
    q_weights: Dict[str, dict],
    input_scale: float,
) -> Dict[str, tuple]:
    """Run float model → compute per-layer calibration params.

    Returns dict mapping layer_name to:
      - Parametric layers (Conv2D, Linear): (requant_m, requant_shift, bias_corrected)
      - Sigmoid: ('sigmoid', scale_mult, scale_shift)
      - Tanh:    ('tanh', scale_mult, scale_shift)
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("PyTorch required") from e

    requant_params: Dict[str, tuple] = {}
    current_scale = input_scale

    with torch.no_grad():
        x = input_tensor.clone()
        for name, module in model.named_children():
            x = module(x)
            class_name = type(module).__name__

            if name in q_weights and q_weights[name]['weights'] is not None:
                # Parametric layer — compute requant + corrected bias
                w_scale = q_weights[name]['weight_scale']
                out_np = x.numpy().flatten()
                out_max = float(np.max(np.abs(out_np)))
                out_scale = out_max / 127.0 if out_max > 0 else 1.0

                m, shift = compute_requant_params(current_scale, w_scale, out_scale)

                bias_corrected = None
                if hasattr(module, 'bias') and module.bias is not None:
                    b_f32 = module.bias.detach().cpu().numpy()
                    bias_corrected = np.clip(
                        np.round(b_f32 / out_scale), -128, 127
                    ).astype(np.int8).flatten().tolist()

                requant_params[name] = (m, shift, bias_corrected)
                current_scale = out_scale

            elif class_name == 'Sigmoid':
                # LUT assumes x/16 → need rescale factor = current_scale * 16
                sm, ss = compute_activation_scale_params(current_scale, 16.0)
                requant_params[name] = ('sigmoid', sm, ss)
                # After sigmoid: output range [0, 1] → scale = 1/127
                current_scale = 1.0 / 127.0

            elif class_name == 'Tanh':
                # LUT assumes x/32 → need rescale factor = current_scale * 32
                sm, ss = compute_activation_scale_params(current_scale, 32.0)
                requant_params[name] = ('tanh', sm, ss)
                # After tanh: output range [-1, 1] → scale = 1/127
                current_scale = 1.0 / 127.0

            # ReLU, Flatten, MaxPool: don't change scale, no params needed

    return requant_params


def quantize_weights(model) -> Dict[str, dict]:
    """Walk a PyTorch model and quantize all parameter tensors.

    Args:
        model: PyTorch nn.Module

    Returns:
        Dict mapping layer_name → {
            'type': str,
            'weights': np.ndarray (i8),
            'bias': np.ndarray (i8),
            'weight_scale': float,
            'bias_scale': float,
            'params': dict (layer-specific params like kernel_size, stride, etc.)
        }
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("PyTorch is required: pip install torch") from e

    result: Dict[str, dict] = {}

    for name, module in model.named_children():
        layer_info: dict = {'type': type(module).__name__, 'params': {}}

        if hasattr(module, 'weight') and module.weight is not None:
            w_np = module.weight.detach().cpu().numpy()
            q_w, w_scale = quantize_to_i8(w_np)
            layer_info['weights'] = q_w
            layer_info['weight_scale'] = w_scale
        else:
            layer_info['weights'] = None
            layer_info['weight_scale'] = 0.0

        if hasattr(module, 'bias') and module.bias is not None:
            b_np = module.bias.detach().cpu().numpy()
            q_b, b_scale = quantize_to_i8(b_np)
            layer_info['bias'] = q_b
            layer_info['bias_scale'] = b_scale
        else:
            layer_info['bias'] = None
            layer_info['bias_scale'] = 0.0

        # Extract layer-specific parameters
        class_name = type(module).__name__
        if class_name == 'Conv2d':
            layer_info['params'] = {
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride[0] if isinstance(module.stride, tuple) else module.stride,
                'padding': module.padding[0] if isinstance(module.padding, tuple) else module.padding,
            }
        elif class_name == 'Linear':
            layer_info['params'] = {
                'in_features': module.in_features,
                'out_features': module.out_features,
            }
        elif class_name == 'MaxPool2d':
            ks = module.kernel_size
            st = module.stride
            layer_info['params'] = {
                'kernel_size': ks if isinstance(ks, int) else ks[0],
                'stride': st if isinstance(st, int) else st[0],
            }

        result[name] = layer_info

    return result


# Mapping from PyTorch layer names to Rust layer builder calls
_LAYER_MAP = {
    'Conv2d': 'FrozenConv2D',
    'Linear': 'FrozenDense',
    'ReLU': 'ReLULayer',
    'Sigmoid': 'SigmoidLayer',
    'Tanh': 'TanhLayer',
    'Flatten': 'FlattenLayer',
    'MaxPool2d': 'MaxPool2DLayer',
}


def export_to_rust(
    model,
    model_name: str = "my_model",
    input_shape: Optional[List[int]] = None,
) -> str:
    """Generate Rust source code from a PyTorch model.

    Args:
        model: PyTorch nn.Module (Sequential)
        model_name: Name for the generated Rust constant
        input_shape: Input shape (e.g., [1, 28, 28] for MNIST)

    Returns:
        Complete Rust source code string
    """
    q_weights = quantize_weights(model)
    lines: List[str] = []

    lines.append("//! Auto-generated by nano_rust_utils.export_to_rust()")
    lines.append("//! DO NOT EDIT — regenerate from Python if model changes.")
    lines.append("")
    lines.append("use nano_rust_core::*;")
    lines.append("")

    # Generate static weight arrays
    for name, info in q_weights.items():
        rust_name = name.upper().replace('.', '_')
        if info['weights'] is not None:
            w_flat = info['weights'].flatten().tolist()
            w_str = ', '.join(str(v) for v in w_flat)
            lines.append(f"static {rust_name}_W: &[i8] = &[{w_str}];")
        if info['bias'] is not None:
            b_flat = info['bias'].flatten().tolist()
            b_str = ', '.join(str(v) for v in b_flat)
            lines.append(f"static {rust_name}_B: &[i8] = &[{b_str}];")
    lines.append("")

    # Generate model construction function
    lines.append(f"/// Build the {model_name} model.")
    lines.append(f"pub fn build_{model_name}() -> NanoResult<()> {{")

    # Generate layer instantiations
    layer_vars: List[str] = []
    for name, info in q_weights.items():
        rust_name = name.upper().replace('.', '_')
        layer_type = info['type']
        var_name = f"layer_{name}"

        if layer_type == 'Conv2d':
            p = info['params']
            lines.append(f"    let {var_name} = FrozenConv2D::new(")
            lines.append(f"        {rust_name}_W, {rust_name}_B,")
            lines.append(f"        {p['in_channels']}, {p['out_channels']},")
            kh = p['kernel_size'][0] if isinstance(p['kernel_size'], tuple) else p['kernel_size']
            kw = p['kernel_size'][1] if isinstance(p['kernel_size'], tuple) else p['kernel_size']
            lines.append(f"        {kh}, {kw}, {p['stride']}, {p['padding']},")
            lines.append(f"    )?;")
            layer_vars.append(var_name)

        elif layer_type == 'Linear':
            p = info['params']
            lines.append(f"    let {var_name} = FrozenDense::new(")
            lines.append(f"        {rust_name}_W, {rust_name}_B,")
            lines.append(f"        {p['in_features']}, {p['out_features']},")
            lines.append(f"    )?;")
            layer_vars.append(var_name)

        elif layer_type == 'ReLU':
            lines.append(f"    let {var_name} = ReLULayer;")
            layer_vars.append(var_name)

        elif layer_type == 'Sigmoid':
            lines.append(f"    let {var_name} = SigmoidLayer;")
            layer_vars.append(var_name)

        elif layer_type == 'Tanh':
            lines.append(f"    let {var_name} = TanhLayer;")
            layer_vars.append(var_name)

        elif layer_type == 'Flatten':
            lines.append(f"    let {var_name} = FlattenLayer;")
            layer_vars.append(var_name)

        elif layer_type == 'MaxPool2d':
            p = info['params']
            lines.append(f"    let {var_name} = MaxPool2DLayer::new(")
            lines.append(f"        {p['kernel_size']}, {p['kernel_size']}, {p['stride']},")
            lines.append(f"    )?;")
            layer_vars.append(var_name)

    # Build the layers array
    refs = ', '.join(f'&{v}' for v in layer_vars)
    lines.append(f"    let layers: &[&dyn Layer] = &[{refs}];")

    # Build model with input shape
    if input_shape:
        if len(input_shape) == 1:
            shape_str = f"Shape::d1({input_shape[0]})"
        elif len(input_shape) == 3:
            shape_str = f"Shape::d3({input_shape[0]}, {input_shape[1]}, {input_shape[2]})"
        else:
            shape_str = f"Shape::d1({sum(input_shape)})"
    else:
        shape_str = "Shape::d1(784) // TODO: set correct input shape"

    lines.append(f"    let model = SequentialModel::new(layers, {shape_str})?;")
    lines.append(f"    Ok(())")
    lines.append(f"}}")

    return '\n'.join(lines)


def export_weights_bin(
    q_weights: Dict[str, dict],
    output_dir: str,
) -> List[str]:
    """Export quantized weights to binary files.

    Each layer gets two files: {name}_w.bin and {name}_b.bin.

    Args:
        q_weights: Output from quantize_weights()
        output_dir: Directory to write .bin files

    Returns:
        List of created file paths
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    created: List[str] = []

    for name, info in q_weights.items():
        if info['weights'] is not None:
            w_file = out_path / f"{name}_w.bin"
            info['weights'].flatten().tofile(w_file)
            created.append(str(w_file))

        if info['bias'] is not None:
            b_file = out_path / f"{name}_b.bin"
            info['bias'].flatten().tofile(b_file)
            created.append(str(b_file))

    return created
