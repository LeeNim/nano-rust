//! Python bindings for nano-rust-core via PyO3.
//!
//! Wraps the core Layer Zoo and SequentialModel for use from Python/Jupyter.
//!
//! Key design: Python doesn't support const generics or `&'static` lifetimes.
//! We use `Box::leak` to promote backbone weights to `'static`, which is safe
//! because the Python object owns the model for its entire lifetime.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use nano_rust_core::{
    Arena, Shape,
    layers::{Layer},
    layers::dense::{FrozenDense, TrainableDense},
    layers::conv::FrozenConv2D,
    layers::activations::{ReLULayer, SigmoidLayer, TanhLayer, SoftmaxLayer, ScaledSigmoidLayer, ScaledTanhLayer},
    layers::flatten::FlattenLayer,
    layers::pooling::MaxPool2DLayer,
    model::SequentialModel,
    math,
};

// =============================================================================
// Helper: convert NanoError to PyValueError
// =============================================================================

fn to_py_err(e: nano_rust_core::NanoError) -> PyErr {
    PyValueError::new_err(format!("{:?}", e))
}

// =============================================================================
// PySequentialModel — the main Python-facing model class
// =============================================================================

/// A sequential neural network model for quantized i8 inference.
///
/// Build by adding layers, then call forward() or predict().
///
/// Example:
///     model = PySequentialModel(input_shape=[1, 28, 28], arena_size=8192)
///     model.add_conv2d(kernel, bias, 1, 8, 3, 3, 1, 1)
///     model.add_relu()
///     model.add_flatten()
///     model.add_dense(weights, bias)
///     result = model.predict(input_data)
#[pyclass]
pub struct PySequentialModel {
    /// Arena buffer (heap-allocated for Python, would be stack on MCU)
    arena_buf: Vec<u8>,
    /// Input shape
    input_shape: Shape,
    /// Store layer objects (owned, leaked to 'static for compatibility)
    /// Each entry: (layer_type, layer_ptr) — we manage lifetimes manually
    layers: Vec<Box<dyn Layer>>,
}

impl PySequentialModel {
    /// Compute total elements of current output shape.
    fn current_output_total(&self) -> PyResult<usize> {
        let mut shape = self.input_shape;
        for layer in self.layers.iter() {
            shape = layer.output_shape(&shape).map_err(to_py_err)?;
        }
        Ok(shape.total())
    }
}

#[pymethods]
impl PySequentialModel {
    /// Create a new empty sequential model.
    ///
    /// Args:
    ///     input_shape: List of ints defining input shape (e.g., [1, 28, 28] for CHW)
    ///     arena_size: Scratch memory size in bytes
    #[new]
    fn new(input_shape: Vec<usize>, arena_size: usize) -> PyResult<Self> {
        let shape = match input_shape.len() {
            1 => Shape::d1(input_shape[0]),
            2 => Shape::d2(input_shape[0], input_shape[1]),
            3 => Shape::d3(input_shape[0], input_shape[1], input_shape[2]),
            4 => Shape::d4(input_shape[0], input_shape[1], input_shape[2], input_shape[3]),
            _ => return Err(PyValueError::new_err("input_shape must have 1-4 dimensions")),
        };

        Ok(Self {
            arena_buf: vec![0u8; arena_size],
            input_shape: shape,
            layers: Vec::new(),
        })
    }

    /// Add a frozen dense layer (weights in Flash simulation).
    fn add_dense(&mut self, weights: Vec<i8>, bias: Vec<i8>) -> PyResult<()> {
        let in_features = self.current_output_total()?;
        let out_features = bias.len();
        let static_weights: &'static [i8] = Box::leak(weights.into_boxed_slice());
        let static_bias: &'static [i8] = Box::leak(bias.into_boxed_slice());
        let layer = FrozenDense::new(static_weights, static_bias, in_features, out_features)
            .map_err(to_py_err)?;
        self.layers.push(Box::new(layer));
        Ok(())
    }

    /// Add a frozen dense layer with calibrated requantization.
    ///
    /// Args:
    ///     weights: Quantized i8 weights [out_features * in_features]
    ///     bias: Quantized i8 bias [out_features]
    ///     requant_m: Fixed-point multiplier (computed from scales)
    ///     requant_shift: Bit shift
    fn add_dense_with_requant(
        &mut self, weights: Vec<i8>, bias: Vec<i8>,
        requant_m: i32, requant_shift: u32,
    ) -> PyResult<()> {
        let in_features = self.current_output_total()?;
        let out_features = bias.len();
        let static_weights: &'static [i8] = Box::leak(weights.into_boxed_slice());
        let static_bias: &'static [i8] = Box::leak(bias.into_boxed_slice());
        let layer = FrozenDense::new_with_requant(
            static_weights, static_bias, in_features, out_features,
            requant_m, requant_shift,
        ).map_err(to_py_err)?;
        self.layers.push(Box::new(layer));
        Ok(())
    }

    /// Add a trainable dense layer (weights in RAM).
    fn add_trainable_dense(&mut self, in_features: usize, out_features: usize) -> PyResult<()> {
        let layer = TrainableDense::new(in_features, out_features);
        self.layers.push(Box::new(layer));
        Ok(())
    }

    /// Add a frozen Conv2D layer.
    fn add_conv2d(
        &mut self,
        kernel: Vec<i8>, bias: Vec<i8>,
        in_ch: usize, out_ch: usize,
        kh: usize, kw: usize,
        stride: usize, padding: usize,
    ) -> PyResult<()> {
        let static_kernel: &'static [i8] = Box::leak(kernel.into_boxed_slice());
        let static_bias: &'static [i8] = Box::leak(bias.into_boxed_slice());
        let layer = FrozenConv2D::new(static_kernel, static_bias, in_ch, out_ch, kh, kw, stride, padding)
            .map_err(to_py_err)?;
        self.layers.push(Box::new(layer));
        Ok(())
    }

    /// Add a frozen Conv2D layer with calibrated requantization.
    fn add_conv2d_with_requant(
        &mut self,
        kernel: Vec<i8>, bias: Vec<i8>,
        in_ch: usize, out_ch: usize,
        kh: usize, kw: usize,
        stride: usize, padding: usize,
        requant_m: i32, requant_shift: u32,
    ) -> PyResult<()> {
        let static_kernel: &'static [i8] = Box::leak(kernel.into_boxed_slice());
        let static_bias: &'static [i8] = Box::leak(bias.into_boxed_slice());
        let layer = FrozenConv2D::new_with_requant(
            static_kernel, static_bias, in_ch, out_ch, kh, kw, stride, padding,
            requant_m, requant_shift,
        ).map_err(to_py_err)?;
        self.layers.push(Box::new(layer));
        Ok(())
    }

    /// Add a ReLU activation layer.
    fn add_relu(&mut self) -> PyResult<()> {
        self.layers.push(Box::new(ReLULayer));
        Ok(())
    }

    /// Add a Sigmoid activation layer.
    fn add_sigmoid(&mut self) -> PyResult<()> {
        self.layers.push(Box::new(SigmoidLayer));
        Ok(())
    }

    /// Add a Tanh activation layer.
    fn add_tanh(&mut self) -> PyResult<()> {
        self.layers.push(Box::new(TanhLayer));
        Ok(())
    }

    /// Add a scale-aware Sigmoid layer.
    /// scale_mult / 2^scale_shift ≈ S_y * 16.0 (S_y = previous layer's output scale)
    fn add_sigmoid_scaled(&mut self, scale_mult: i32, scale_shift: u32) -> PyResult<()> {
        self.layers.push(Box::new(ScaledSigmoidLayer { scale_mult, scale_shift }));
        Ok(())
    }

    /// Add a scale-aware Tanh layer.
    /// scale_mult / 2^scale_shift ≈ S_y * 32.0 (S_y = previous layer's output scale)
    fn add_tanh_scaled(&mut self, scale_mult: i32, scale_shift: u32) -> PyResult<()> {
        self.layers.push(Box::new(ScaledTanhLayer { scale_mult, scale_shift }));
        Ok(())
    }

    /// Add a Softmax activation layer.
    fn add_softmax(&mut self) -> PyResult<()> {
        self.layers.push(Box::new(SoftmaxLayer));
        Ok(())
    }

    /// Add a Flatten layer.
    fn add_flatten(&mut self) -> PyResult<()> {
        self.layers.push(Box::new(FlattenLayer));
        Ok(())
    }

    /// Add a MaxPool2D layer.
    fn add_max_pool2d(&mut self, pool_h: usize, pool_w: usize, stride: usize) -> PyResult<()> {
        let layer = MaxPool2DLayer::new(pool_h, pool_w, stride).map_err(to_py_err)?;
        self.layers.push(Box::new(layer));
        Ok(())
    }

    /// Run forward pass, return output as list of i8 values.
    fn forward(&mut self, input: Vec<i8>) -> PyResult<Vec<i8>> {
        let mut arena = Arena::new(&mut self.arena_buf);

        // Build layer refs for SequentialModel
        let layer_refs: Vec<&dyn Layer> = self.layers.iter().map(|l| l.as_ref() as &dyn Layer).collect();
        let model = SequentialModel::new(&layer_refs, self.input_shape).map_err(to_py_err)?;

        let (output, _shape) = model.forward(&input, &mut arena).map_err(to_py_err)?;
        let result = output.to_vec();
        Ok(result)
    }

    /// Predict class index (forward + argmax).
    fn predict(&mut self, input: Vec<i8>) -> PyResult<usize> {
        let mut arena = Arena::new(&mut self.arena_buf);

        let layer_refs: Vec<&dyn Layer> = self.layers.iter().map(|l| l.as_ref() as &dyn Layer).collect();
        let model = SequentialModel::new(&layer_refs, self.input_shape).map_err(to_py_err)?;

        model.predict(&input, &mut arena).map_err(to_py_err)
    }

    /// Get output shape as a list.
    fn output_shape(&self) -> PyResult<Vec<usize>> {
        let mut shape = self.input_shape;
        for layer in self.layers.iter() {
            shape = layer.output_shape(&shape).map_err(to_py_err)?;
        }
        Ok(shape.dims[..shape.ndim].to_vec())
    }

    /// Get estimated arena size needed.
    fn estimate_arena_size(&self) -> PyResult<usize> {
        let layer_refs: Vec<&dyn Layer> = self.layers.iter().map(|l| l.as_ref() as &dyn Layer).collect();
        let model = SequentialModel::new(&layer_refs, self.input_shape).map_err(to_py_err)?;
        model.estimate_arena_size().map_err(to_py_err)
    }

    /// Get number of layers.
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get layer names as a list of strings.
    fn layer_names(&self) -> Vec<String> {
        self.layers.iter().map(|l| l.name().to_string()).collect()
    }

    /// Export model configuration to binary format.
    ///
    /// Format: "NANO" magic + layer descriptors + weight data.
    fn export_to_flash(&self, path: &str) -> PyResult<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)
            .map_err(|e| PyValueError::new_err(format!("Failed to create file: {}", e)))?;

        // Magic header
        file.write_all(b"NANO").map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        // Number of layers
        let num = self.layers.len() as u32;
        file.write_all(&num.to_le_bytes()).map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        // Write layer names for reconstruction
        for layer in self.layers.iter() {
            let name = layer.name();
            let name_len = name.len() as u16;
            file.write_all(&name_len.to_le_bytes()).map_err(|e| PyValueError::new_err(format!("{}", e)))?;
            file.write_all(name.as_bytes()).map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        }

        Ok(())
    }
}

/// Python module initialization.
#[pymodule]
fn nano_rust_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySequentialModel>()?;
    Ok(())
}
