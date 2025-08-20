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
import ai.djl.inference.streaming.StreamingTranslator.StreamOutput;
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;

@SuppressWarnings("PMD.TestClassWithoutTestCases")
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

            try (Predictor<Double, Iterator<Double>> predictor =
                    model.newPredictor(new TestTranslator())) {
                Assert.assertTrue(predictor.supportsStreaming());

                // Test iterative streaming
                List<Double> results = new ArrayList<>(3);
                predictor.streamingPredict(1.0).getIterativeOutput().forEachRemaining(results::add);
                Assert.assertEquals(results, Arrays.asList(1.0, 1.0));

                // Test async streaming
                StreamOutput<Iterator<Double>> so = predictor.streamingPredict(1.0);
                results = new ArrayList<>(3);
                Iterator<Double> id = so.getAsyncOutput();
                so.computeAsyncOutput();
                id.forEachRemaining(results::add);
                Assert.assertEquals(results, Arrays.asList(1.0, 1.0));
            }
        }
    }

    private static final class TestTranslator
            implements StreamingTranslator<Double, Iterator<Double>> {

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, Double input) {
            return new NDList(ctx.getNDManager().create(input));
        }

        /** {@inheritDoc} */
        @Override
        public Iterator<Double> processOutput(TranslatorContext ctx, NDList list) {
            return Arrays.stream(list.singletonOrThrow().toDoubleArray()).iterator();
        }

        @Override
        public StreamOutput<Iterator<Double>> processStreamOutput(
                TranslatorContext ctx, Stream<NDList> list) {
            return new StreamOutput<>() {
                List<Double> outList;

                @Override
                protected Iterator<Double> buildAsyncOutput() {
                    outList = new ArrayList<>();
                    return new TestListIterator<>(outList);
                }

                @Override
                protected void computeAsyncOutputInternal(Iterator<Double> output) {
                    list.forEach(
                            d -> {
                                outList.add(d.singletonOrThrow().getDouble());
                            });
                }

                @Override
                public Iterator<Double> getIterativeOutputInternal() {
                    return list.mapToDouble(l -> l.singletonOrThrow().getDouble()).iterator();
                }
            };
        }

        /** {@inheritDoc} */
        @Override
        public Support getSupport() {
            return Support.BOTH;
        }
    }

    /**
     * A testing {@link Iterator} for a list with concurrent modification.
     *
     * @param <E> the list and iterator element type
     */
    private static class TestListIterator<E> implements Iterator<E> {

        List<E> lst;
        int index;

        public TestListIterator(List<E> lst) {
            this.lst = lst;
        }

        @Override
        public boolean hasNext() {
            return lst.size() > index;
        }

        @Override
        public E next() {
            return lst.get(index++);
        }
    }
}
