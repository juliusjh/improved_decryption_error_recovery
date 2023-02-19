mod bin_tree;
mod bin_var_node;
mod check_graph;
mod check_msg;
mod check_nodes;
mod variable_node;

use crate::check_graph::{__pyo3_get_function_test_fft_2, __pyo3_get_function_test_fft_3};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
#[cfg(feature = "kyber1024")]
fn check_bp1024(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<check_graph::CheckGraph>()?;
    m.add_class::<bin_tree::PyBinMultTreeInt>()?;
    m.add_class::<bin_tree::PyBinMultTreeList>()?;
    m.add_function(wrap_pyfunction!(test_fft_2, m)?).unwrap();
    m.add_function(wrap_pyfunction!(test_fft_3, m)?).unwrap();

    Ok(())
}

#[pymodule]
#[cfg(feature = "kyber768")]
fn check_bp768(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<check_graph::CheckGraph>()?;
    m.add_class::<bin_tree::PyBinMultTreeInt>()?;
    m.add_class::<bin_tree::PyBinMultTreeList>()?;
    m.add_function(wrap_pyfunction!(test_fft_2, m)?).unwrap();
    m.add_function(wrap_pyfunction!(test_fft_3, m)?).unwrap();

    Ok(())
}

#[pymodule]
#[cfg(feature = "kyber512")]
fn check_bp512(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<check_graph::CheckGraph>()?;
    m.add_class::<bin_tree::PyBinMultTreeInt>()?;
    m.add_class::<bin_tree::PyBinMultTreeList>()?;
    m.add_function(wrap_pyfunction!(test_fft_2, m)?).unwrap();
    m.add_function(wrap_pyfunction!(test_fft_3, m)?).unwrap();

    Ok(())
}

#[pymodule]
#[cfg(feature = "frodo640")]
fn check_bp640(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<check_graph::CheckGraph>()?;
    m.add_class::<bin_tree::PyBinMultTreeInt>()?;
    m.add_class::<bin_tree::PyBinMultTreeList>()?;
    m.add_function(wrap_pyfunction!(test_fft_2, m)?).unwrap();
    m.add_function(wrap_pyfunction!(test_fft_3, m)?).unwrap();

    Ok(())
}
