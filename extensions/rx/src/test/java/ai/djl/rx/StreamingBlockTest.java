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
package ai.djl.rx;

import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter.Type;
import ai.djl.nn.core.Linear;
import ai.djl.training.initializer.Initializer;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class StreamingBlockTest {

    @Test
    public void testSequential() throws TranslateException {
        try (Model model = Model.newInstance("test")) {
            StreamingSequentialBlock block = new StreamingSequentialBlock();
            block.add(Linear.builder().setUnits(1).build());
            block.add(Linear.builder().setUnits(1).build());
            model.setBlock(block);

            block.setInitializer(Initializer.ONES, Type.WEIGHT);
            block.initialize(model.getNDManager(), DataType.FLOAT32, new Shape(1, 1));

            try (StreamingPredictor<Float, Float> predictor =
                    new StreamingPredictor<>(
                            model, new TestTranslator(), model.getNDManager().getDevice(), false)) {
                List<Float> results =
                        predictor
                                .streamingPredict(1f)
                                .blockingStream()
                                .collect(Collectors.toList());
                Assert.assertEquals(results, Arrays.asList(1f, 1f));
            }
        }
    }

    private static class TestTranslator implements Translator<Float, Float> {

        /** {@inheritDoc} */
        @Override
        public Float processOutput(TranslatorContext ctx, NDList list) throws Exception {
            return list.singletonOrThrow().getFloat();
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, Float input) throws Exception {
            return new NDList(ctx.getNDManager().create(input));
        }
    }
}
