// Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

extern crate tokenizers as tk;

use std::borrow::Borrow;
use std::str::FromStr;
use tk::tokenizer::{EncodeInput, Encoding};
use tk::{FromPretrainedParameters, Offsets};
use tk::Tokenizer;

use jni::objects::{JClass, JMethodID, JObject, JString, JValue};
use jni::sys::{jboolean, jlong, jlongArray, jobjectArray, jsize, JNI_TRUE, jintArray, jint, jvalue};
use jni::JNIEnv;

#[no_mangle]
pub extern "system" fn Java_ai_djl_huggingface_tokenizers_jni_TokenizersLibrary_createTokenizer(
    env: JNIEnv,
    _: JObject,
    input: JString,
) -> jlong {
    let identifier: String = env
        .get_string(input)
        .expect("Couldn't get java string!")
        .into();

    let parameters = FromPretrainedParameters::default();
    let tokenizer = Tokenizer::from_pretrained(identifier, Some(parameters));

    match tokenizer {
        Ok(output) => to_handle(output),
        Err(err) => {
            env.throw(err.to_string()).unwrap();
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_huggingface_tokenizers_jni_TokenizersLibrary_createTokenizerFromString(
    env: JNIEnv,
    _: JObject,
    json: JString,
) -> jlong {
    let data: String = env
        .get_string(json)
        .expect("Couldn't get java string!")
        .into();

    let tokenizer = Tokenizer::from_str(&data);
    match tokenizer {
        Ok(output) => to_handle(output),
        Err(err) => {
            env.throw(err.to_string()).unwrap();
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_huggingface_tokenizers_jni_TokenizersLibrary_deleteTokenizer(
    _env: JNIEnv,
    _: JObject,
    handle: jlong,
) {
    drop_handle::<Tokenizer>(handle);
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_huggingface_tokenizers_jni_TokenizersLibrary_encode(
    env: JNIEnv,
    _: JObject,
    handle: jlong,
    input: JString,
    add_special_tokens: jboolean,
) -> jlong {
    let tokenizer = cast_handle::<Tokenizer>(handle);
    let sequence: String = env
        .get_string(input)
        .expect("Couldn't get java string!")
        .into();

    let input_sequence = tk::InputSequence::from(sequence);
    let encoded_input = EncodeInput::Single(input_sequence);
    let encoding = tokenizer.encode(encoded_input, add_special_tokens == JNI_TRUE);

    match encoding {
        Ok(output) => to_handle(output),
        Err(err) => {
            env.throw(err.to_string()).unwrap();
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_huggingface_tokenizers_jni_TokenizersLibrary_encodeList(
    env: JNIEnv,
    _: JObject,
    handle: jlong,
    inputs: jobjectArray,
    add_special_tokens: jboolean,
) -> jlong {
    let tokenizer = cast_handle::<Tokenizer>(handle);
    let len = env.get_array_length(inputs).unwrap();
    let mut array: Vec<String> = Vec::new();
    for i in 0..len {
        let item = env.get_object_array_element(inputs, i).unwrap().into();
        let value: String = env
            .get_string(item)
            .expect("Couldn't get java string!")
            .into();
        array.push(value);
    }

    let input_sequence = tk::InputSequence::from(array);
    let encoded_input = EncodeInput::from(input_sequence);
    let encoding = tokenizer.encode(encoded_input, add_special_tokens == JNI_TRUE);

    match encoding {
        Ok(output) => to_handle(output),
        Err(err) => {
            env.throw(err.to_string()).unwrap();
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_huggingface_tokenizers_jni_TokenizersLibrary_batchEncode(
    env: JNIEnv,
    _: JObject,
    handle: jlong,
    inputs: jobjectArray,
    add_special_tokens: jboolean,
) -> jlongArray {
    let tokenizer = cast_handle::<Tokenizer>(handle);
    let len = env.get_array_length(inputs).unwrap();
    let mut array: Vec<String> = Vec::new();
    for i in 0..len {
        let item = env.get_object_array_element(inputs, i).unwrap().into();
        let value: String = env
            .get_string(item)
            .expect("Couldn't get java string!")
            .into();
        array.push(value);
    }

    let encodings = tokenizer
        .encode_batch(array, add_special_tokens == JNI_TRUE)
        .unwrap();
    let handles = encodings
        .into_iter()
        .map(|c| to_handle(c))
        .collect::<Vec<_>>();

    let size = handles.len() as jsize;
    let ret = env.new_long_array(size).unwrap();
    env.set_long_array_region(ret, 0, &handles).unwrap();
    ret
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_huggingface_tokenizers_jni_TokenizersLibrary_deleteEncoding(
    _env: JNIEnv,
    _: JObject,
    handle: jlong,
) {
    drop_handle::<Encoding>(handle);
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_huggingface_tokenizers_jni_TokenizersLibrary_getTokenIds(
    env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlongArray {
    let encoding = cast_handle::<Encoding>(handle);
    let ids = encoding.get_ids();
    let len = ids.len() as jsize;
    let mut long_ids: Vec<jlong> = Vec::new();
    long_ids.reserve(len as usize);
    for i in ids {
        long_ids.push(*i as jlong)
    }

    let array: jlongArray = env.new_long_array(len).unwrap();
    env.set_long_array_region(array, 0, &long_ids).unwrap();
    array
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_huggingface_tokenizers_jni_TokenizersLibrary_getTypeIds(
    env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlongArray {
    let encoding = cast_handle::<Encoding>(handle);
    let type_ids = encoding.get_type_ids();
    let len = type_ids.len() as jsize;
    let mut long_ids: Vec<jlong> = Vec::new();
    for i in type_ids {
        long_ids.push(*i as jlong)
    }

    let array: jlongArray = env.new_long_array(len).unwrap();
    env.set_long_array_region(array, 0, &long_ids).unwrap();
    array
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_huggingface_tokenizers_jni_TokenizersLibrary_getWordIds(
    env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlongArray {
    let encoding = cast_handle::<Encoding>(handle);
    let word_ids = encoding.get_word_ids();
    let len = word_ids.len() as jsize;
    let mut long_ids: Vec<jlong> = Vec::new();
    for i in word_ids {
        if let Some(word_id) = i {
            long_ids.push(*word_id as jlong)
        } else {
            long_ids.push(-1)
        }
    }

    let array: jlongArray = env.new_long_array(len).unwrap();
    env.set_long_array_region(array, 0, &long_ids).unwrap();
    array
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_huggingface_tokenizers_jni_TokenizersLibrary_getTokens(
    env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jobjectArray {
    let encoding = cast_handle::<Encoding>(handle);
    let tokens = encoding.get_tokens();
    let len = tokens.len() as jsize;

    let array: jobjectArray = env
        .new_object_array(len, "java/lang/String", JObject::null())
        .unwrap();
    for (i, token) in tokens.iter().enumerate() {
        let item: JString = env.new_string(&token).unwrap();
        env.set_object_array_element(array, i as jsize, item).unwrap();
    }
    array
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_huggingface_tokenizers_jni_TokenizersLibrary_getAttentionMask(
    env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlongArray {
    let encoding = cast_handle::<Encoding>(handle);
    let attention_masks = encoding.get_attention_mask();
    let len = attention_masks.len() as jsize;
    let mut long_ids: Vec<jlong> = Vec::new();
    for i in attention_masks {
        long_ids.push(*i as jlong)
    }

    let array: jlongArray = env.new_long_array(len).unwrap();
    env.set_long_array_region(array, 0, &long_ids).unwrap();
    array
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_huggingface_tokenizers_jni_TokenizersLibrary_getSpecialTokenMask(
    env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlongArray {
    let encoding = cast_handle::<Encoding>(handle);
    let special_token_masks = encoding.get_special_tokens_mask();
    let len = special_token_masks.len() as jsize;
    let mut long_ids: Vec<jlong> = Vec::new();
    for i in special_token_masks {
        long_ids.push(*i as jlong)
    }

    let array: jlongArray = env.new_long_array(len).unwrap();
    env.set_long_array_region(array, 0, &long_ids).unwrap();
    array
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_huggingface_tokenizers_jni_TokenizersLibrary_getTokenCharSpans(
    env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jobjectArray {
    let encoding = cast_handle::<Encoding>(handle);
    let tokens = encoding.get_tokens();
    let len = tokens.len() as jsize;

    let array: jobjectArray = env
        .new_object_array(len, "ai/djl/huggingface/tokenizers/jni/CharSpan", JObject::null())
        .unwrap();
    for (i, _) in tokens.iter().enumerate() {
        let opt_offsets: Option<(usize, Offsets)> = encoding.token_to_chars(i);
        //let spans: Option<(usize, Offsets)> = encoding.token_to_chars(i);
        //maybe check if rust has something like getOrElse?
        match &opt_offsets {
            Some((_, offsets)) => {
                println!("char spans: {:?}", offsets);
               // offsets_vec.push((*offsets).0 as jint);
               // offsets_vec.push((*offsets).1 as jint);
                let classId = "ai/djl/huggingface/tokenizers/CharSpan";
                let methodId = "<init>";
                let params = "(II)V";
                let cls: JClass = env.find_class(classId).unwrap();
                let constructor: JMethodID = env.get_method_id(cls, methodId, params).unwrap();
                let offsets_vec: Vec<JValue> = vec![JValue::Int((*offsets).0 as jint), JValue::Int((*offsets).0 as jint) ];
                let obj = env.new_object_unchecked(cls, constructor, &offsets_vec[..] ).unwrap();
                env.set_object_array_element(array, i as jsize, obj).unwrap();
            }
            None => {
                //set Object null here
                println!("no char spans!");
                //no spans, set as empty array
                // let item: jintArray = env.new_int_array(0).unwrap();
                // env.set_int_array_region(item, 0, &offsets_vec).unwrap();
                // env.set_object_array_element(array, i as jsize, item).unwrap();
            }
        }

        }
    array
}

fn to_handle<T: 'static>(val: T) -> jlong {
    let handle = Box::into_raw(Box::new(val)) as jlong;
    handle
}

fn cast_handle<T>(handle: jlong) -> &'static mut T {
    assert_ne!(handle, 0, "Invalid handle value");

    let ptr = handle as *mut T;
    unsafe { &mut *ptr }
}

fn drop_handle<T: 'static>(handle: jlong) {
    unsafe {
        Box::from_raw(handle as *mut T);
    }
}
