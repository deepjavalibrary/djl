/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.onnxruntime.zoo.tabular.softmax_regression;

import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** A {@link TranslatorFactory} that creates a {@link IrisTranslator} instance. */
public class IrisClassificationTranslatorFactory implements TranslatorFactory {

    /** {@inheritDoc} */
    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        return Collections.singleton(new Pair<>(IrisFlower.class, Classifications.class));
    }

    /** {@inheritDoc} */
    @Override
    public Translator<?, ?> newInstance(
            Class<?> input, Class<?> output, Model model, Map<String, ?> arguments) {
        if (!isSupported(input, output)) {
            throw new IllegalArgumentException("Unsupported input/output types.");
        }
        return new IrisTranslator();
    }

    private static final class IrisTranslator
            implements NoBatchifyTranslator<IrisFlower, Classifications> {

        private List<String> synset;

        public IrisTranslator() {
            // species name
            synset = Arrays.asList("setosa", "versicolor", "virginica");
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, IrisFlower input) {
            float[] data = {
                input.getSepalLength(),
                input.getSepalWidth(),
                input.getPetalLength(),
                input.getPetalWidth()
            };
            NDArray array = ctx.getNDManager().create(data, new Shape(1, 4));
            return new NDList(array);
        }

        /** {@inheritDoc} */
        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            float[] data = list.get(1).toFloatArray();
            List<Double> probabilities = new ArrayList<>(data.length);
            for (float f : data) {
                probabilities.add((double) f);
            }
            return new Classifications(synset, probabilities);
        }
    }
}
