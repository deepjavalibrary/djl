use crate::layers::HiddenAct;
use candle::{Device, Result, Tensor};
use std::sync::Once;

#[cfg(feature = "cuda")]
use candle_cublaslt::{fused_batch_matmul, fused_matmul, Activation, CublasLt};

pub fn get_cublas_lt_wrapper(device: &Device) -> Option<CublasLtWrapper> {
    unsafe {
        let cublaslt = if device.is_cuda() {
            #[cfg(feature = "cuda")]
            {
                Some(CublasLtWrapper {
                    cublaslt: CublasLt::new(&device).unwrap(),
                })
            }
            #[cfg(not(feature = "cuda"))]
            {
                None
            }
        } else {
            None
        };
        cublaslt
    }
}

#[derive(Debug, Clone)]
pub struct CublasLtWrapper {
    #[cfg(feature = "cuda")]
    pub cublaslt: CublasLt,
}

impl CublasLtWrapper {
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
        out: Option<&Tensor>,
        alpha: Option<f32>,
        beta: Option<f32>,
        bias: Option<&Tensor>,
        act: Option<HiddenAct>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let inner_act = match act {
                Some(HiddenAct::Gelu) => Some(Activation::Gelu),
                Some(HiddenAct::Relu) => Some(Activation::Relu),
                _ => None,
            };

            let mut result = fused_matmul(
                a,
                b,
                out,
                alpha,
                beta,
                bias,
                inner_act,
                self.cublaslt.clone(),
            )?;

            if Some(HiddenAct::Swiglu) == act {
                result = candle_nn::ops::swiglu(&result)?;
            }
            Ok(result)
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle::bail!("`cuda` feature is not enabled")
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
        out: Option<&Tensor>,
        alpha: Option<f32>,
        beta: Option<f32>,
        bias: Option<&Tensor>,
        act: Option<HiddenAct>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let inner_act = match act {
                Some(HiddenAct::Gelu) => Some(Activation::Gelu),
                Some(HiddenAct::Relu) => Some(Activation::Relu),
                _ => None,
            };

            let mut result = fused_batch_matmul(
                a,
                b,
                out,
                alpha,
                beta,
                bias,
                inner_act,
                self.cublaslt.clone(),
            )?;

            if Some(HiddenAct::Swiglu) == act {
                result = candle_nn::ops::swiglu(&result)?;
            }
            Ok(result)
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle::bail!("`cuda` feature is not enabled")
        }
    }
}
