/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.llama.jni;

import java.util.Map;

/** A class holds llama.cpp model loading parameters. */
@SuppressWarnings("PMD.SingularField")
public final class ModelParameters {

    private int nThreads;
    private int nCtx;
    private int nBatch;
    private int nGpuLayers;
    private int mainGpu;
    private int numa;
    private float ropeFreqBase;
    private float ropeFreqScale;
    private boolean mulMatQ;
    private boolean f16Kv;
    private boolean logitsAll;
    private boolean vocabOnly;
    private boolean useMmap;
    private boolean useMlock;
    private boolean embedding;
    private boolean memoryF16;
    private boolean memTest;
    private boolean verbosePrompt;
    private float[] tensorSplit;
    private String loraAdapter;
    private String loraBase;

    /**
     * Constructs a new {@code ModelParameters} instance.
     *
     * @param options the model loading options
     */
    public ModelParameters(Map<String, ?> options) {
        nThreads = intValue(options, "number_threads", Runtime.getRuntime().availableProcessors());
        nCtx = intValue(options, "max_context_length", 512);
        nBatch = intValue(options, "max_rolling_batch", 512);
        nGpuLayers = intValue(options, "number_gpu_layers", -1);
        mainGpu = intValue(options, "tensor_parallel_degree", 0);
        ropeFreqBase = floatValue(options, "rope_freq_base");
        ropeFreqScale = floatValue(options, "ropeFreqScale");
        f16Kv = booleanValue(options, "f16_kv");
        // unused since llamaCPP commit 3ab8b3a, left for backward compatibility, has no effect.
        mulMatQ = booleanValue(options, "mulmat_q", true);
        logitsAll = booleanValue(options, "logits_all");
        vocabOnly = booleanValue(options, "vocab_only");
        useMmap = booleanValue(options, "use_mmap", true);
        useMlock = booleanValue(options, "use_mlock");
        embedding = booleanValue(options, "embedding");
        memoryF16 = booleanValue(options, "memory_f16", true);
        memTest = booleanValue(options, "mem_test");
        verbosePrompt = booleanValue(options, "verbose_prompt");
        numa = numaValue(options, "numa");
        String val = stringValue(options, "tensor_split");
        if (val != null && !val.isEmpty()) {
            String[] tokens = val.split(",");
            tensorSplit = new float[tokens.length];
            for (int i = 0; i < tokens.length; ++i) {
                tensorSplit[i] = Float.parseFloat(tokens[i].trim());
            }
        }
        loraAdapter = stringValue(options, "lora_adapter");
        loraBase = stringValue(options, "loraBase");
    }

    private static int intValue(Map<String, ?> arguments, String key, int def) {
        Object value = arguments.get(key);
        if (value == null) {
            return def;
        }
        return (int) Double.parseDouble(value.toString());
    }

    private static float floatValue(Map<String, ?> arguments, String key) {
        Object value = arguments.get(key);
        if (value == null) {
            return 0f;
        }
        return (float) Double.parseDouble(value.toString());
    }

    private static boolean booleanValue(Map<String, ?> arguments, String key) {
        return booleanValue(arguments, key, false);
    }

    private static boolean booleanValue(Map<String, ?> arguments, String key, boolean def) {
        Object value = arguments.get(key);
        if (value == null) {
            return def;
        }
        return Boolean.parseBoolean(value.toString());
    }

    private static String stringValue(Map<String, ?> arguments, String key) {
        Object value = arguments.get(key);
        if (value == null) {
            return null;
        }
        return value.toString();
    }

    private static int numaValue(Map<String, ?> arguments, String key) {
        /* "disabled" -> 0, "distribute" -> 1, "isolate" -> 2, "numactl" -> 3, "mirror" -> 4 */
        Object value = arguments.get(key);
        if (value == null) {
            return 0;
        }
        if (value.toString().contains("disabled") || value.toString().contains("false")) {
            return 0;
        }
        if (value.toString().contains("distribute") || value.toString().contains("true")) {
            /* 1 for backward compatibility ? */
            return 1;
        }
        if (value.toString().contains("isolate")) {
            return 2;
        }
        if (value.toString().contains("numactl")) {
            return 3;
        }
        if (value.toString().contains("mirror")) {
            return 4;
        }
        return 0;
    }
}
