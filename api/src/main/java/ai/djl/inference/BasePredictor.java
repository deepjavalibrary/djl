/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.inference;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class BasePredictor<I, O> implements Predictor<I, O> {

    private Translator<I, O> translator;
    private long timestamp;

    protected Model model;
    protected NDManager manager;
    Metrics metrics;
    private Block block;
    private ParameterStore parameterStore;

    public BasePredictor(Model model, Translator<I, O> translator) {
        this.model = model;
        this.manager = model.getNDManager().newSubManager();
        this.translator = translator;
        block = model.getBlock();
        parameterStore = new ParameterStore(manager, true);
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("PMD.AvoidRethrowingException")
    public O predict(I input) throws TranslateException {
        return batchPredict(Collections.singletonList(input)).get(0);
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("PMD.AvoidRethrowingException")
    public List<O> batchPredict(List<I> inputs) throws TranslateException {
        try (PredictorContext inputCtx = new PredictorContext();
                PredictorContext outputCtx = new PredictorContext()) {

            Batchifier batchifier = translator.getBatchifier();
            if (batchifier == null) {
                List<O> ret = new ArrayList<>(inputs.size());
                for (I input : inputs) {
                    timestamp = System.nanoTime();
                    NDList ndList = translator.processInput(inputCtx, input);
                    preprocessEnd();

                    NDList result = forward(ndList);
                    forwardEnd(result);

                    ret.add(translator.processOutput(outputCtx, result));
                    postProcessEnd();
                }
                return ret;
            }

            timestamp = System.nanoTime();
            NDList inputBatch = processInputs(inputCtx, inputs);
            preprocessEnd();

            NDList result = forward(inputBatch);
            forwardEnd(result);

            return processOutputs(outputCtx, result);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new TranslateException(e);
        } finally {
            postProcessEnd();
        }
    }

    /** {@inheritDoc} */
    @Override
    public void setMetrics(Metrics metrics) {
        this.metrics = metrics;
    }

    protected void waitToRead(NDList list) {}

    protected NDList forward(NDList ndList) {
        return block.forward(parameterStore, ndList);
    }

    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    private NDList processInputs(TranslatorContext ctx, List<I> inputs) throws Exception {
        int batchSize = inputs.size();
        NDList[] preprocessed = new NDList[batchSize];
        for (int i = 0; i < batchSize; ++i) {
            preprocessed[i] = translator.processInput(ctx, inputs.get(i));
        }
        return translator.getBatchifier().batchify(preprocessed);
    }

    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    private List<O> processOutputs(TranslatorContext ctx, NDList list) throws Exception {
        NDList[] unbatched = translator.getBatchifier().unbatchify(list);
        List<O> outputs = new ArrayList<>(unbatched.length);
        for (NDList output : unbatched) {
            outputs.add(translator.processOutput(ctx, output));
        }
        return outputs;
    }

    private void preprocessEnd() {
        if (metrics != null) {
            long tmp = System.nanoTime();
            long duration = tmp - timestamp;
            timestamp = tmp;
            metrics.addMetric("Preprocess", duration, "nano");
        }
    }

    private void forwardEnd(NDList list) {
        if (metrics != null) {
            waitToRead(list);
            long tmp = System.nanoTime();
            long duration = tmp - timestamp;
            timestamp = tmp;
            metrics.addMetric("Inference", duration, "nano");
        }
    }

    private void postProcessEnd() {
        if (metrics != null) {
            long tmp = System.nanoTime();
            long duration = tmp - timestamp;
            timestamp = tmp;
            metrics.addMetric("Postprocess", duration, "nano");
        }
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        manager.close();
    }

    private class PredictorContext implements TranslatorContext {

        private NDManager ctxManager;

        PredictorContext() {
            ctxManager = manager.newSubManager();
        }

        /** {@inheritDoc} */
        @Override
        public Model getModel() {
            return model;
        }

        /** {@inheritDoc} */
        @Override
        public NDManager getNDManager() {
            return ctxManager;
        }

        /** {@inheritDoc} */
        @Override
        public Metrics getMetrics() {
            return metrics;
        }

        /** {@inheritDoc} */
        @Override
        public void close() {
            ctxManager.close();
        }
    }
}
