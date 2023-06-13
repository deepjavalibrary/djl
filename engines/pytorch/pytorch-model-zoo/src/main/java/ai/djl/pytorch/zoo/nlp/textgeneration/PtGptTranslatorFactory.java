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
package ai.djl.pytorch.zoo.nlp.textgeneration;

import ai.djl.Model;
import ai.djl.modality.nlp.generate.CausalLMOutput;
import ai.djl.ndarray.NDList;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;

import java.lang.reflect.Type;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class PtGptTranslatorFactory implements TranslatorFactory {

    private static final Set<Pair<Type, Type>> SUPPORTED_TYPES = new HashSet<>();

    static {
        SUPPORTED_TYPES.add(new Pair<>(NDList.class, CausalLMOutput.class));
    }

    /** {@inheritDoc} */
    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        return SUPPORTED_TYPES;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public <I, O> Translator<I, O> newInstance(
            Class<I> input, Class<O> output, Model model, Map<String, ?> arguments) {
        if (!isSupported(input, output)) {
            throw new IllegalArgumentException("Unsupported input/output types.");
        }
        long kvDim = ArgumentsUtil.longValue(arguments, "kvDim", 64);
        int numAttentionHeads = ArgumentsUtil.intValue(arguments, "numAttentionHeads", 12);
        int numLayers = ArgumentsUtil.intValue(arguments, "numLayers", 12);

        return (Translator<I, O>) (new PtGptTranslator(kvDim, numAttentionHeads, numLayers));
    }
}
