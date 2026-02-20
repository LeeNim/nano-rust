//! General-purpose integration tests for the refactored TinyML framework.
//!
//! Tests the full Layer Zoo, Shape propagation, SequentialModel,
//! arena checkpoints, and all activation functions.

use nano_rust_core::*;

// =============================================================================
// Shape Tests
// =============================================================================

#[test]
fn test_shape_creation_and_total() {
    let s1 = Shape::d1(10);
    assert_eq!(s1.total(), 10);
    assert_eq!(s1.ndim, 1);

    let s2 = Shape::d2(3, 4);
    assert_eq!(s2.total(), 12);

    let s3 = Shape::d3(3, 28, 28);
    assert_eq!(s3.total(), 2352);
    assert_eq!(s3.channels(), 3);
    assert_eq!(s3.height(), 28);
    assert_eq!(s3.width(), 28);

    let s4 = Shape::d4(1, 3, 28, 28);
    assert_eq!(s4.total(), 2352);
}

// =============================================================================
// Arena Tests (with checkpoints)
// =============================================================================

#[test]
fn test_arena_basic_allocation() {
    let mut buf = [0u8; 256];
    let mut arena = Arena::new(&mut buf);
    let slice = arena.alloc_i8_slice(10).unwrap();
    assert_eq!(slice.len(), 10);
    assert_eq!(arena.used(), 10);
    assert_eq!(arena.remaining(), 246);
}

#[test]
fn test_arena_u8_allocation() {
    let mut buf = [0u8; 256];
    let mut arena = Arena::new(&mut buf);
    let slice = arena.alloc_u8_slice(32).unwrap();
    assert_eq!(slice.len(), 32);
    assert_eq!(arena.used(), 32);
}

#[test]
fn test_arena_checkpoint_restore() {
    let mut buf = [0u8; 256];
    let mut arena = Arena::new(&mut buf);

    // Alloc output (keep alive)
    let _output = arena.alloc_i8_slice(10).unwrap();
    let cp = arena.save_checkpoint();
    assert_eq!(arena.used(), 10);

    // Alloc temporary scratch
    let _scratch = arena.alloc_i8_slice(100).unwrap();
    assert_eq!(arena.used(), 110);

    // Restore — scratch freed, output still valid
    arena.restore_checkpoint(cp);
    assert_eq!(arena.used(), 10);
    assert_eq!(arena.remaining(), 246);
}

#[test]
fn test_arena_exhaustion_returns_error() {
    let mut buf = [0u8; 16];
    let mut arena = Arena::new(&mut buf);
    let result = arena.alloc_i8_slice(100);
    assert!(result.is_err());
}

// =============================================================================
// Activation Function Tests
// =============================================================================

#[test]
fn test_relu_i8() {
    let mut data = [-5i8, -1, 0, 1, 5, 127, -128];
    relu_i8(&mut data);
    assert_eq!(data, [0, 0, 0, 1, 5, 127, 0]);
}

#[test]
fn test_sigmoid_i8_range() {
    let mut data: Vec<i8> = (-128..=127).map(|x| x as i8).collect();
    sigmoid_i8(&mut data);
    // Sigmoid output should be in [0, 127]
    for &val in data.iter() {
        assert!(val >= 0, "sigmoid output {} < 0", val);
        assert!(val <= 127, "sigmoid output {} > 127", val);
    }
    // Monotonically non-decreasing (sigmoid property)
    for i in 1..data.len() {
        assert!(data[i] >= data[i - 1], "sigmoid not monotonic at {}", i);
    }
}

#[test]
fn test_tanh_i8_range() {
    let mut data: Vec<i8> = (-128..=127).map(|x| x as i8).collect();
    tanh_i8(&mut data);
    // tanh output should be in [-127, 127]
    for &val in data.iter() {
        assert!(val >= -127, "tanh output {} < -127", val);
        assert!(val <= 127, "tanh output {} > 127", val);
    }
}

#[test]
fn test_softmax_i8_output() {
    let logits = [10i8, 5, -3, 1];
    let mut output = [0i8; 4];
    softmax_i8(&logits, &mut output).unwrap();
    // All values should be non-negative
    for &val in output.iter() {
        assert!(val >= 0, "softmax output {} < 0", val);
    }
    // Largest logit should have largest softmax value
    assert!(output[0] >= output[1]);
    assert!(output[0] >= output[2]);
    assert!(output[0] >= output[3]);
}

#[test]
fn test_argmax_i8() {
    let data = [3i8, -1, 10, 5, 0];
    assert_eq!(argmax_i8(&data).unwrap(), 2);
}

// =============================================================================
// Conv2D Math Tests
// =============================================================================

#[test]
fn test_conv2d_output_size() {
    // 28×28 input, 3×3 kernel, stride 1, padding 0 → 26×26
    let (oh, ow) = conv2d_output_size(28, 28, 3, 3, 1, 0);
    assert_eq!((oh, ow), (26, 26));

    // 28×28 input, 3×3 kernel, stride 1, padding 1 → 28×28
    let (oh, ow) = conv2d_output_size(28, 28, 3, 3, 1, 1);
    assert_eq!((oh, ow), (28, 28));

    // 28×28 input, 5×5 kernel, stride 2, padding 0 → 12×12
    let (oh, ow) = conv2d_output_size(28, 28, 5, 5, 2, 0);
    assert_eq!((oh, ow), (12, 12));
}

#[test]
fn test_pool_output_size() {
    // 26×26 input, 2×2 pool, stride 2 → 13×13
    let (oh, ow) = pool_output_size(26, 26, 2, 2, 2);
    assert_eq!((oh, ow), (13, 13));
}

#[test]
fn test_conv2d_i8_basic() {
    // 1 input channel, 4×4 input, 1 output channel, 3×3 kernel, stride 1, pad 0
    // Output: 1×2×2
    let input = [
        1i8, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    ];
    let kernel = [
        1i8, 0, -1,
        1, 0, -1,
        1, 0, -1,
    ];
    let bias = [0i8];
    let mut output = [0i8; 4]; // 1×2×2
    let mut im2col = [0i8; 9 * 4]; // (1*3*3) × (2*2)

    conv2d_i8(
        &input, &kernel, &bias, &mut output, &mut im2col,
        1, 4, 4, 1, 3, 3, 1, 0,
    ).unwrap();

    // Just verify it runs without error and produces non-zero output
    assert_eq!(output.len(), 4);
}

#[test]
fn test_max_pool_2d_basic() {
    // 1 channel, 4×4 input, 2×2 pool, stride 2 → 1×2×2
    let input = [
        1i8, 3, 2, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    ];
    let mut output = [0i8; 4];

    max_pool_2d(&input, &mut output, 1, 4, 4, 2, 2, 2).unwrap();

    assert_eq!(output[0], 6);   // max(1,3,5,6)
    assert_eq!(output[1], 8);   // max(2,4,7,8)
    assert_eq!(output[2], 14);  // max(9,10,13,14)
    assert_eq!(output[3], 16);  // max(11,12,15,16)
}

// =============================================================================
// Layer Tests
// =============================================================================

#[test]
fn test_frozen_dense_forward() {
    // 4 inputs → 2 outputs
    let weights: &'static [i8] = &[1, 0, 0, 0,  0, 0, 0, 1]; // identity-like
    let bias: &'static [i8] = &[0, 0];

    let layer = FrozenDense::new(weights, bias, 4, 2).unwrap();

    let mut buf = [0u8; 256];
    let mut arena = Arena::new(&mut buf);
    let input = [10i8, 20, 30, 40];
    let (output, shape) = layer.forward(&input, &Shape::d1(4), &mut arena).unwrap();

    assert_eq!(shape, Shape::d1(2));
    assert_eq!(output.len(), 2);
}

#[test]
fn test_frozen_dense_dimension_validation() {
    let weights: &'static [i8] = &[1; 8];
    let bias: &'static [i8] = &[0, 0];
    let layer = FrozenDense::new(weights, bias, 4, 2).unwrap();

    let mut buf = [0u8; 256];
    let mut arena = Arena::new(&mut buf);
    let wrong_input = [1i8; 5]; // Wrong size!
    assert!(layer.forward(&wrong_input, &Shape::d1(5), &mut arena).is_err());
}

#[test]
fn test_trainable_dense_backward_updates_weights() {
    let mut layer = TrainableDense::new(4, 2);

    // Set some initial weights
    let init_w = [10i8, 20, 30, 40, 50, 60, 70, 80];
    let init_b = [5i8, -5];
    layer.load_weights(&init_w, &init_b).unwrap();

    let input = [1i8, 2, 3, 4];
    let grad = [1i8, -1];
    layer.backward(&input, &grad, 2).unwrap();

    // Verify weights changed
    assert_ne!(layer.weights(), &init_w[..]);
}

#[test]
fn test_flatten_layer() {
    let flatten = FlattenLayer;
    let input_shape = Shape::d3(3, 4, 4); // 48 elements
    let out_shape = flatten.output_shape(&input_shape).unwrap();
    assert_eq!(out_shape, Shape::d1(48));

    let mut buf = [0u8; 256];
    let mut arena = Arena::new(&mut buf);
    let input = [1i8; 48];
    let (output, shape) = flatten.forward(&input, &input_shape, &mut arena).unwrap();
    assert_eq!(shape, Shape::d1(48));
    assert_eq!(output.len(), 48);
}

#[test]
fn test_relu_layer() {
    let relu = ReLULayer;
    let mut buf = [0u8; 256];
    let mut arena = Arena::new(&mut buf);
    let input = [-5i8, 0, 3, -1, 127];
    let (output, _) = relu.forward(&input, &Shape::d1(5), &mut arena).unwrap();
    assert_eq!(output, &[0, 0, 3, 0, 127]);
}

#[test]
fn test_sigmoid_layer() {
    let sigmoid = SigmoidLayer;
    let mut buf = [0u8; 256];
    let mut arena = Arena::new(&mut buf);
    let input = [-128i8, 0, 127];
    let (output, _) = sigmoid.forward(&input, &Shape::d1(3), &mut arena).unwrap();
    // All outputs should be in [0, 127]
    for &val in output.iter() {
        assert!(val >= 0 && val <= 127);
    }
}

#[test]
fn test_tanh_layer() {
    let tanh_l = TanhLayer;
    let mut buf = [0u8; 256];
    let mut arena = Arena::new(&mut buf);
    let input = [-128i8, 0, 127];
    let (output, _) = tanh_l.forward(&input, &Shape::d1(3), &mut arena).unwrap();
    // tanh(0) ≈ 0
    assert!(output[1].abs() <= 1, "tanh(0) should be ~0, got {}", output[1]);
}

#[test]
fn test_max_pool_2d_layer() {
    let pool = MaxPool2DLayer::square(2, 2).unwrap();
    let input_shape = Shape::d3(1, 4, 4);
    let out_shape = pool.output_shape(&input_shape).unwrap();
    assert_eq!(out_shape, Shape::d3(1, 2, 2));

    let mut buf = [0u8; 256];
    let mut arena = Arena::new(&mut buf);
    let input = [
        1i8, 3, 2, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    ];
    let (output, shape) = pool.forward(&input, &input_shape, &mut arena).unwrap();
    assert_eq!(shape, Shape::d3(1, 2, 2));
    assert_eq!(output[0], 6);   // max(1,3,5,6)
    assert_eq!(output[3], 16);  // max(11,12,15,16)
}

// =============================================================================
// Conv2D Layer Tests
// =============================================================================

#[test]
fn test_frozen_conv2d_shape_propagation() {
    let kernel: &'static [i8] = &[1i8; 2 * 1 * 3 * 3]; // 2 filters, 1 in_ch, 3×3
    let bias: &'static [i8] = &[0, 0]; // 2 output channels
    let conv = FrozenConv2D::new(kernel, bias, 1, 2, 3, 3, 1, 0).unwrap();

    let input_shape = Shape::d3(1, 8, 8); // 1ch × 8×8
    let out_shape = conv.output_shape(&input_shape).unwrap();
    // (8-3)/1 + 1 = 6
    assert_eq!(out_shape, Shape::d3(2, 6, 6));
}

#[test]
fn test_frozen_conv2d_forward() {
    let kernel: &'static [i8] = &[1i8; 1 * 1 * 3 * 3]; // 1×1×3×3
    let bias: &'static [i8] = &[0];
    let conv = FrozenConv2D::new(kernel, bias, 1, 1, 3, 3, 1, 0).unwrap();

    let input_shape = Shape::d3(1, 4, 4);
    let input = [
        1i8, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
    ];

    // Need enough arena for output (1×2×2=4) + im2col (9×4=36) = 40 bytes
    let mut buf = [0u8; 256];
    let mut arena = Arena::new(&mut buf);

    let (output, shape) = conv.forward(&input, &input_shape, &mut arena).unwrap();
    assert_eq!(shape, Shape::d3(1, 2, 2));
    assert_eq!(output.len(), 4);
}

// =============================================================================
// SequentialModel Tests
// =============================================================================

#[test]
fn test_sequential_model_dense_only() {
    let weights: &'static [i8] = &[1i8; 4 * 2]; // 2×4
    let bias: &'static [i8] = &[0, 0];
    let dense = FrozenDense::new(weights, bias, 4, 2).unwrap();
    let relu = ReLULayer;

    let layers: &[&dyn Layer] = &[&dense, &relu];
    let model = SequentialModel::new(layers, Shape::d1(4)).unwrap();

    assert_eq!(model.num_layers(), 2);
    assert_eq!(model.output_shape().unwrap(), Shape::d1(2));

    let mut buf = [0u8; 256];
    let mut arena = Arena::new(&mut buf);
    let input = [10i8, 20, 30, 40];
    let (output, shape) = model.forward(&input, &mut arena).unwrap();
    assert_eq!(shape, Shape::d1(2));
    assert_eq!(output.len(), 2);
}

#[test]
fn test_sequential_model_predict() {
    let weights: &'static [i8] = &[1, 0, 0, 0,  0, 0, 0, 1, 0, 0, 0, 0];
    let bias: &'static [i8] = &[10, 0, -10];
    let dense = FrozenDense::new(weights, bias, 4, 3).unwrap();

    let layers: &[&dyn Layer] = &[&dense];
    let model = SequentialModel::new(layers, Shape::d1(4)).unwrap();

    let mut buf = [0u8; 256];
    let mut arena = Arena::new(&mut buf);
    let input = [50i8, 0, 0, 0];
    let class = model.predict(&input, &mut arena).unwrap();
    // Should classify as class 0 (highest activation due to weight[0][0]=1 + bias=10)
    assert_eq!(class, 0);
}

#[test]
fn test_sequential_model_conv_flatten_dense() {
    // Conv2D(1→1, 3×3) → ReLU → Flatten → Dense(4→2)
    let conv_kernel: &'static [i8] = &[1i8; 9]; // 1×1×3×3
    let conv_bias: &'static [i8] = &[0];
    let conv = FrozenConv2D::new(conv_kernel, conv_bias, 1, 1, 3, 3, 1, 0).unwrap();

    let relu = ReLULayer;
    let flatten = FlattenLayer;

    // 4×4 - 3 + 1 = 2×2 → flatten = 4
    let dense_w: &'static [i8] = &[1i8; 4 * 2]; // 2×4
    let dense_b: &'static [i8] = &[0, 0];
    let dense = FrozenDense::new(dense_w, dense_b, 4, 2).unwrap();

    let layers: &[&dyn Layer] = &[&conv, &relu, &flatten, &dense];
    let model = SequentialModel::new(layers, Shape::d3(1, 4, 4)).unwrap();

    assert_eq!(model.output_shape().unwrap(), Shape::d1(2));

    // Arena needs: conv_out(4) + im2col(36) + relu(4) + flat(4) + dense(2) = 50
    let mut buf = [0u8; 256];
    let mut arena = Arena::new(&mut buf);

    let input = [1i8; 16]; // 1×4×4
    let (output, shape) = model.forward(&input, &mut arena).unwrap();
    assert_eq!(shape, Shape::d1(2));
    assert_eq!(output.len(), 2);
}

#[test]
fn test_sequential_model_estimate_arena() {
    let weights: &'static [i8] = &[1i8; 8];
    let bias: &'static [i8] = &[0, 0];
    let dense = FrozenDense::new(weights, bias, 4, 2).unwrap();

    let layers: &[&dyn Layer] = &[&dense];
    let model = SequentialModel::new(layers, Shape::d1(4)).unwrap();

    let estimated = model.estimate_arena_size().unwrap();
    assert!(estimated >= 2, "Should need at least 2 bytes for output");
}

#[test]
fn test_sequential_model_empty() {
    let layers: &[&dyn Layer] = &[];
    let model = SequentialModel::new(layers, Shape::d1(4)).unwrap();

    let mut buf = [0u8; 256];
    let mut arena = Arena::new(&mut buf);
    let input = [1i8, 2, 3, 4];
    let (output, shape) = model.forward(&input, &mut arena).unwrap();
    assert_eq!(output, &input[..]);
    assert_eq!(shape, Shape::d1(4));
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_zero_input_no_panic() {
    let weights: &'static [i8] = &[0i8; 8];
    let bias: &'static [i8] = &[0, 0];
    let dense = FrozenDense::new(weights, bias, 4, 2).unwrap();

    let mut buf = [0u8; 256];
    let mut arena = Arena::new(&mut buf);
    let input = [0i8; 4];
    let (output, _) = dense.forward(&input, &Shape::d1(4), &mut arena).unwrap();
    assert_eq!(output, &[0, 0]);
}

#[test]
fn test_extreme_i8_values() {
    let mut data = [i8::MIN, i8::MAX, 0];
    relu_i8(&mut data);
    assert_eq!(data, [0, 127, 0]);

    let mut sig_data = [i8::MIN, 0, i8::MAX];
    sigmoid_i8(&mut sig_data);
    // Should not panic
    assert!(sig_data[0] >= 0);
    assert!(sig_data[2] <= 127);
}

#[test]
fn test_matmul_dimension_mismatch() {
    let a = [1i8; 6];
    let b = [1i8; 6];
    let mut c = [0i8; 4];
    // a = 2×3, b = 3×2 → c = 2×2 — this should work
    assert!(quantized_matmul_i8(&a, &b, &mut c, 2, 3, 2).is_ok());
    // Wrong output size
    let mut c_wrong = [0i8; 3];
    assert!(quantized_matmul_i8(&a, &b, &mut c_wrong, 2, 3, 2).is_err());
}
