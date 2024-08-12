mod bert;
mod camembert;
mod distilbert;
mod roberta;
mod xlm_roberta;

use crate::ndarray::as_data_type;
use crate::{cast_handle, drop_handle, to_handle, to_string_array};
use bert::{BertConfig, BertForSequenceClassification, BertModel};
use camembert::{CamembertConfig, CamembertModel};
use candle::{DType, Device, Error, Result, Tensor};
use candle_nn::VarBuilder;
use distilbert::{DistilBertConfig, DistilBertModel};
use jni::objects::{JLongArray, JObject, JString, ReleaseMode};
use jni::sys::{jint, jlong, jobjectArray};
use jni::JNIEnv;
use roberta::{RobertaConfig, RobertaForSequenceClassification, RobertaModel};
use serde::Deserialize;
use std::path::PathBuf;
use xlm_roberta::{XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaModel};

#[derive(Debug, PartialEq, Clone)]
#[allow(dead_code, unused)]
pub enum Pool {
    Cls,
    Mean,
    Splade,
    LastToken,
}

#[derive(Deserialize)]
#[serde(tag = "model_type", rename_all = "kebab-case")]
enum Config {
    Bert(BertConfig),
    Camembert(CamembertConfig),
    Roberta(RobertaConfig),
    XlmRoberta(XLMRobertaConfig),
    #[serde(rename(deserialize = "distilbert"))]
    DistilBert(DistilBertConfig),
}

pub(crate) trait Model {
    fn get_input_names(&self) -> Vec<String>;

    fn forward(
        &self,
        _input_ids: &Tensor,
        _attention_mask: &Tensor,
        _token_type_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        candle::bail!("`forward` is not implemented for this model");
    }
}

fn load_model<'local>(
    env: &mut JNIEnv,
    model_path: JString,
    dtype: jint,
) -> Result<Box<dyn Model>> {
    let model_path: String = env
        .get_string(&model_path)
        .expect("Couldn't get java string!")
        .into();

    let model_path = PathBuf::from(model_path);

    // Load config
    let config: String = std::fs::read_to_string(model_path.join("config.json"))?;
    let config: Config = serde_json::from_str(&config).map_err(Error::msg)?;

    // Get candle device
    let device = if candle::utils::cuda_is_available() {
        Device::new_cuda(0)
    } else if candle::utils::metal_is_available() {
        Device::new_metal(0)
    } else {
        Ok(Device::Cpu)
    }?;

    // Get candle dtype
    let dtype = as_data_type(dtype).unwrap();

    let safetensors_path = model_path.join("model.safetensors");
    let vb = if safetensors_path.exists() {
        unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_path.join("model.safetensors")],
                dtype,
                &device,
            )?
        }
    } else {
        VarBuilder::from_pth(model_path.join("pytorch_model.bin"), dtype, &device)?
    };

    let use_flash_attn = cfg!(feature = "cuda")
        && cfg!(feature = "flash-attn")
        && dtype == DType::F16
        && std::env::var("USE_FLASH_ATTENTION")
            .ok()
            .map_or(true, |v| v.parse().unwrap_or(true));

    let model: Result<Box<dyn Model>> = match (config, &device) {
        #[cfg(not(feature = "cuda"))]
        (_, Device::Cuda(_)) => panic!("`cuda` feature is not enabled"),
        (Config::Bert(mut config), _) => {
            tracing::info!("Starting Bert model on {:?}", device);
            config.use_flash_attn = Some(use_flash_attn);
            match config.architectures.first() {
                Some(arch) => match arch.as_str() {
                    "BertForSequenceClassification" => {
                        Ok(Box::new(BertForSequenceClassification::load(vb, &config)?))
                    }
                    _ => Ok(Box::new(BertModel::load(vb, &config)?)),
                },
                None => Ok(Box::new(BertModel::load(vb, &config)?)),
            }
        }
        (Config::Camembert(mut config), _) => {
            tracing::info!("Starting Camembert model on {:?}", device);
            config.use_flash_attn = Some(use_flash_attn);
            Ok(Box::new(CamembertModel::load(vb, &config)?))
        }
        (Config::Roberta(mut config), _) => {
            tracing::info!("Starting Roberta model on {:?}", device);
            config.use_flash_attn = Some(use_flash_attn);
            match config.architectures.first() {
                Some(arch) => match arch.as_str() {
                    "RobertaForSequenceClassification" => Ok(Box::new(
                        RobertaForSequenceClassification::load(vb, &config)?,
                    )),
                    _ => Ok(Box::new(RobertaModel::load(vb, &config)?)),
                },
                None => Ok(Box::new(RobertaModel::load(vb, &config)?)),
            }
        }
        (Config::XlmRoberta(mut config), _) => {
            tracing::info!("Starting XlmRoberta model on {:?}", device);
            config.use_flash_attn = Some(use_flash_attn);
            match config.architectures.first() {
                Some(arch) => match arch.as_str() {
                    "XLMRobertaForSequenceClassification" => Ok(Box::new(
                        XLMRobertaForSequenceClassification::load(vb, &config)?,
                    )),
                    _ => Ok(Box::new(XLMRobertaModel::load(vb, &config)?)),
                },
                None => Ok(Box::new(XLMRobertaModel::load(vb, &config)?)),
            }
        }
        (Config::DistilBert(mut config), _) => {
            tracing::info!("Starting DistilBert model on {:?}", device);
            config.use_flash_attn = Some(use_flash_attn);
            Ok(Box::new(DistilBertModel::load(vb, &config)?))
        }
    };

    model
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_loadModel<'local>(
    mut env: JNIEnv,
    _: JObject,
    model_path: JString,
    dtype: jint,
) -> jlong {
    let model = load_model(&mut env, model_path, dtype);

    match model {
        Ok(output) => to_handle(output),
        Err(err) => {
            env.throw(err.to_string()).unwrap();
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_deleteModel<'local>(
    _: JNIEnv,
    _: JObject,
    handle: jlong,
) {
    drop_handle::<Box<dyn Model>>(handle);
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_getInputNames<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jobjectArray {
    let model = cast_handle::<Box<dyn Model>>(handle);
    let input_names: Vec<String> = model.get_input_names();
    to_string_array(&mut env, input_names).unwrap()
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_runInference<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    input_handles: JLongArray<'local>,
) -> jlong {
    let model = cast_handle::<Box<dyn Model>>(handle);
    let input_handles =
        unsafe { env.get_array_elements(&input_handles, ReleaseMode::NoCopyBack) }.unwrap();

    let mut input_vec: Vec<&Tensor> = Vec::new();
    for &i in input_handles.iter() {
        let tensor = cast_handle::<Tensor>(i);
        input_vec.push(tensor);
    }

    let result = model.forward(
        input_vec.get(0).unwrap(),
        input_vec.get(1).unwrap(),
        input_vec.get(2).map(|&x| x),
    );

    match result {
        Ok(output) => to_handle(output),
        Err(err) => {
            env.throw(err.to_string()).unwrap();
            0
        }
    }
}
