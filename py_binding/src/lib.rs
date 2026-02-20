//! Python bindings for nano-rust-core via PyO3.

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

fn to_py_err(e: nano_rust_core::NanoError) -> PyErr {
    PyValueError::new_err(format!("{:?}", e))
}

/// A sequential neural network model for quantized i8 inference.
#[pyclass]
pub struct PySequentialModel {
    arena_buf: Vec<u8>,
    input_shape: Shape,
    layers: Vec<Box<dyn Layer>>,
}

impl PySequentialModel {
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
    #[new]
    fn new(input_shape: Vec<usize>, arena_size: usize) -> PyResult<Self> {
        let shape = match input_shape.len() {
            1 => Shape::d1(input_shape[0]),
            2 => Shape::d2(input_shape[0], input_shape[1]),
            3 => Shape::d3(input_shape[0], input_shape[1], input_shape[2]),
            4 => Shape::d4(input_shape[0], input_shape[1], input_shape[2], input_shape[3]),
            _ => return Err(PyValueError::new_err("input_shape must have 1-4 dimensions")),
        };
        Ok(Self { arena_buf: vec![0u8; arena_size], input_shape: shape, layers: Vec::new() })
    }

    fn add_dense(&mut self, weights: Vec<i8>, bias: Vec<i8>) -> PyResult<()> {
        let in_features = self.current_output_total()?;
        let out_features = bias.len();
        let static_weights: &'static [i8] = Box::leak(weights.into_boxed_slice());
        let static_bias: &'static [i8] = Box::leak(bias.into_boxed_slice());
        let layer = FrozenDense::new(static_weights, static_bias, in_features, out_features).map_err(to_py_err)?;
        self.layers.push(Box::new(layer));
        Ok(())
    }

    fn add_dense_with_requant(&mut self, weights: Vec<i8>, bias: Vec<i8>, requant_m: i32, requant_shift: u32) -> PyResult<()> {
        let in_features = self.current_output_total()?;
        let out_features = bias.len();
        let static_weights: &'static [i8] = Box::leak(weights.into_boxed_slice());
        let static_bias: &'static [i8] = Box::leak(bias.into_boxed_slice());
        let layer = FrozenDense::new_with_requant(static_weights, static_bias, in_features, out_features, requant_m, requant_shift).map_err(to_py_err)?;
        self.layers.push(Box::new(layer));
        Ok(())
    }

    fn add_trainable_dense(&mut self, in_features: usize, out_features: usize) -> PyResult<()> {
        let layer = TrainableDense::new(in_features, out_features);
        self.layers.push(Box::new(layer));
        Ok(())
    }

    fn add_conv2d(&mut self, kernel: Vec<i8>, bias: Vec<i8>, in_ch: usize, out_ch: usize, kh: usize, kw: usize, stride: usize, padding: usize) -> PyResult<()> {
        let static_kernel: &'static [i8] = Box::leak(kernel.into_boxed_slice());
        let static_bias: &'static [i8] = Box::leak(bias.into_boxed_slice());
        let layer = FrozenConv2D::new(static_kernel, static_bias, in_ch, out_ch, kh, kw, stride, padding).map_err(to_py_err)?;
        self.layers.push(Box::new(layer));
        Ok(())
    }

    fn add_conv2d_with_requant(&mut self, kernel: Vec<i8>, bias: Vec<i8>, in_ch: usize, out_ch: usize, kh: usize, kw: usize, stride: usize, padding: usize, requant_m: i32, requant_shift: u32) -> PyResult<()> {
        let static_kernel: &'static [i8] = Box::leak(kernel.into_boxed_slice());
        let static_bias: &'static [i8] = Box::leak(bias.into_boxed_slice());
        let layer = FrozenConv2D::new_with_requant(static_kernel, static_bias, in_ch, out_ch, kh, kw, stride, padding, requant_m, requant_shift).map_err(to_py_err)?;
        self.layers.push(Box::new(layer));
        Ok(())
    }

    fn add_relu(&mut self) -> PyResult<()> { self.layers.push(Box::new(ReLULayer)); Ok(()) }
    fn add_sigmoid(&mut self) -> PyResult<()> { self.layers.push(Box::new(SigmoidLayer)); Ok(()) }
    fn add_tanh(&mut self) -> PyResult<()> { self.layers.push(Box::new(TanhLayer)); Ok(()) }
    fn add_sigmoid_scaled(&mut self, scale_mult: i32, scale_shift: u32) -> PyResult<()> {
        self.layers.push(Box::new(ScaledSigmoidLayer { scale_mult, scale_shift })); Ok(())
    }
    fn add_tanh_scaled(&mut self, scale_mult: i32, scale_shift: u32) -> PyResult<()> {
        self.layers.push(Box::new(ScaledTanhLayer { scale_mult, scale_shift })); Ok(())
    }
    fn add_softmax(&mut self) -> PyResult<()> { self.layers.push(Box::new(SoftmaxLayer)); Ok(()) }
    fn add_flatten(&mut self) -> PyResult<()> { self.layers.push(Box::new(FlattenLayer)); Ok(()) }

    fn add_max_pool2d(&mut self, pool_h: usize, pool_w: usize, stride: usize) -> PyResult<()> {
        let layer = MaxPool2DLayer::new(pool_h, pool_w, stride).map_err(to_py_err)?;
        self.layers.push(Box::new(layer));
        Ok(())
    }

    fn forward(&mut self, input: Vec<i8>) -> PyResult<Vec<i8>> {
        let mut arena = Arena::new(&mut self.arena_buf);
        let layer_refs: Vec<&dyn Layer> = self.layers.iter().map(|l| l.as_ref() as &dyn Layer).collect();
        let model = SequentialModel::new(&layer_refs, self.input_shape).map_err(to_py_err)?;
        let (output, _shape) = model.forward(&input, &mut arena).map_err(to_py_err)?;
        let result = output.to_vec();
        Ok(result)
    }

    fn predict(&mut self, input: Vec<i8>) -> PyResult<usize> {
        let mut arena = Arena::new(&mut self.arena_buf);
        let layer_refs: Vec<&dyn Layer> = self.layers.iter().map(|l| l.as_ref() as &dyn Layer).collect();
        let model = SequentialModel::new(&layer_refs, self.input_shape).map_err(to_py_err)?;
        model.predict(&input, &mut arena).map_err(to_py_err)
    }

    fn output_shape(&self) -> PyResult<Vec<usize>> {
        let mut shape = self.input_shape;
        for layer in self.layers.iter() { shape = layer.output_shape(&shape).map_err(to_py_err)?; }
        Ok(shape.dims[..shape.ndim].to_vec())
    }

    fn estimate_arena_size(&self) -> PyResult<usize> {
        let layer_refs: Vec<&dyn Layer> = self.layers.iter().map(|l| l.as_ref() as &dyn Layer).collect();
        let model = SequentialModel::new(&layer_refs, self.input_shape).map_err(to_py_err)?;
        model.estimate_arena_size().map_err(to_py_err)
    }

    fn num_layers(&self) -> usize { self.layers.len() }
    fn layer_names(&self) -> Vec<String> { self.layers.iter().map(|l| l.name().to_string()).collect() }

    fn export_to_flash(&self, path: &str) -> PyResult<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path).map_err(|e| PyValueError::new_err(format!("Failed to create file: {}", e)))?;
        file.write_all(b"NANO").map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        let num = self.layers.len() as u32;
        file.write_all(&num.to_le_bytes()).map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        for layer in self.layers.iter() {
            let name = layer.name();
            let name_len = name.len() as u16;
            file.write_all(&name_len.to_le_bytes()).map_err(|e| PyValueError::new_err(format!("{}", e)))?;
            file.write_all(name.as_bytes()).map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        }
        Ok(())
    }
}

#[pymodule]
fn nano_rust_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySequentialModel>()?;
    Ok(())
}
