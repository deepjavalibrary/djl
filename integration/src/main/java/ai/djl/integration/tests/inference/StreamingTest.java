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
package ai.djl.integration.tests.inference;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.inference.streaming.StreamingTranslator;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter.Type;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.initializer.Initializer;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

public class StreamingTest {

    @Test
    public void testSequential() throws TranslateException {
        try (Model model = Model.newInstance("test")) {
            SequentialBlock block = new SequentialBlock();
            block.add(Linear.builder().setUnits(1).build());
            block.add(Linear.builder().setUnits(1).build());
            model.setBlock(block);

            block.setInitializer(Initializer.ONES, Type.WEIGHT);
            block.initialize(model.getNDManager(), DataType.FLOAT64, new Shape(1, 1));

            try (Predictor<Double, DoubleStream> predictor =
                    model.newPredictor(new TestTranslator())) {
                Assert.assertTrue(predictor.supportsStreaming());
                List<Double> results =
                        predictor.streamingPredict(1.0).boxed().collect(Collectors.toList());
                Assert.assertEquals(results, Arrays.asList(1.0, 1.0));
            }
        }
    }

    private static class TestTranslator implements StreamingTranslator<Double, DoubleStream> {

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, Double input) {
            return new NDList(ctx.getNDManager().create(input));
        }

        /** {@inheritDoc} */
        @Override
        public DoubleStream processOutput(TranslatorContext ctx, NDList list) {
            return Arrays.stream(list.singletonOrThrow().toDoubleArray());
        }

        @Override
        public DoubleStream processStreamOutput(TranslatorContext ctx, Stream<NDList> list) {
            return list.mapToDouble(l -> l.singletonOrThrow().getDouble());
        }
    }
}
