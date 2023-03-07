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

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import io.reactivex.rxjava3.core.Flowable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A {@link Predictor} that adds support for streaming predictions.
 *
 * @param <I> input type
 * @param <O> output type
 * @see #streamingPredict(Object)
 */
public class StreamingPredictor<I, O> extends Predictor<I, List<O>> {

    StreamingBlock streamingBlock;
    Translator<I, O> baseTranslator;

    /**
     * Creates a new instance of {@code BasePredictor} with the given {@link Model} and {@link
     * Translator}.
     *
     * @param model the model on which the predictions are based
     * @param translator the translator to be used
     * @param device the device for prediction
     * @param copy whether to copy the parameters to the parameter store. If the device changes, it
     *     will copy regardless
     */
    public StreamingPredictor(
            Model model, Translator<I, O> translator, Device device, boolean copy) {
        super(model, new ListMapTranslator<>(translator), device, copy);
        baseTranslator = translator;

        if (!(model.getBlock() instanceof StreamingBlock)) {
            throw new IllegalArgumentException(
                    "Expected a StreamingBlock to the StreamingPredictor");
        }
        streamingBlock = (StreamingBlock) model.getBlock();
    }

    /**
     * Predicts an item for inference.
     *
     * @param input the input
     * @return the output object defined by the user
     * @throws TranslateException if an error occurs during prediction
     */
    @SuppressWarnings({"PMD.AvoidRethrowingException", "PMD.IdenticalCatchBranches"})
    public Flowable<O> streamingPredict(I input) throws TranslateException {
        try {
            PredictorContext context = newPredictorContext();
            if (!prepared) {
                baseTranslator.prepare(context);
                prepared = true;
            }
            Batchifier batchifier = baseTranslator.getBatchifier();
            if (batchifier == null) {
                NDList ndList = baseTranslator.processInput(context, input);

                return streamingBlock
                        .forwardStream(parameterStore, ndList, false)
                        .map(result -> baseTranslator.processOutput(context, result))
                        .doAfterTerminate(context::close);
            }

            // For the batched case, need to create singleton batch and unbatchify singleton
            NDList inputBatch = processInputs(context, Collections.singletonList(input));
            return streamingBlock
                    .forwardStream(parameterStore, inputBatch, false)
                    .map(
                            result -> {
                                NDList[] unbatched =
                                        baseTranslator.getBatchifier().unbatchify(result);
                                if (unbatched.length != 1) {
                                    throw new IllegalStateException(
                                            "Unexpected number of outputs from model");
                                }
                                return baseTranslator.processOutput(context, unbatched[0]);
                            })
                    .doAfterTerminate(context::close);

        } catch (TranslateException e) {
            throw e;
        } catch (Exception e) {
            throw new TranslateException(e);
        }
    }

    private static final class ListMapTranslator<I, O> implements Translator<I, List<O>> {

        Translator<I, O> baseTranslator;

        private ListMapTranslator(Translator<I, O> baseTranslator) {
            this.baseTranslator = baseTranslator;
        }

        /** {@inheritDoc} */
        @Override
        public List<O> processOutput(TranslatorContext ctx, NDList list) throws Exception {
            NDList[] split = Batchifier.STACK.unbatchify(list);
            List<O> outputs = new ArrayList<>(split.length);
            for (NDList s : split) {
                outputs.add(baseTranslator.processOutput(ctx, s));
            }
            return outputs;
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, I input) throws Exception {
            return baseTranslator.processInput(ctx, input);
        }

        /** {@inheritDoc} */
        @Override
        public void prepare(TranslatorContext ctx) throws Exception {
            baseTranslator.prepare(ctx);
        }
    }
}
