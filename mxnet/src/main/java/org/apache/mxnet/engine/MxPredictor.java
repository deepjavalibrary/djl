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
package org.apache.mxnet.engine;

import com.amazon.ai.Context;
import com.amazon.ai.Model;
import com.amazon.ai.TranslateException;
import com.amazon.ai.Translator;
import com.amazon.ai.TranslatorContext;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.metric.Metric;
import com.amazon.ai.metric.Metrics;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.util.Pair;
import org.apache.mxnet.jna.JnaUtils;

public class MxPredictor<I, O> implements Predictor<I, O> {

    MxModel model;
    private Translator<I, O> translator;
    Context context;
    private CachedOp cachedOp;
    MxNDFactory factory;
    Metrics metrics;
    private long timestamp;

    MxPredictor(MxModel model, Translator<I, O> translator, Context context) {
        this.factory = MxNDFactory.SYSTEM_FACTORY.newSubFactory(context);
        this.model = model;
        this.translator = translator;
        this.context = context;
        cachedOp = JnaUtils.createCachedOp(model, factory);
        metrics = new Metrics();
    }

    /** {@inheritDoc} */
    @Override
    public O predict(I input) throws TranslateException {
        timestamp = System.nanoTime();

        try (PredictorContext inputCtx = new PredictorContext();
                PredictorContext outputCtx = new PredictorContext()) {
            NDList ndList = translator.processInput(inputCtx, input);
            preprocessEnd();

            NDList result = forward(ndList);
            forwardEnd(result);

            return translator.processOutput(outputCtx, result);
        } finally {
            postProcessEnd();
        }
    }

    /** {@inheritDoc} */
    @Override
    public void setMetrics(Metrics metrics) {
        this.metrics = metrics;
    }

    private NDList forward(NDList ndList) {
        return cachedOp.forward(ndList);
    }

    private void preprocessEnd() {
        if (metrics != null) {
            long tmp = System.nanoTime();
            long duration = tmp - timestamp;
            timestamp = tmp;
            metrics.addMetric(new Metric("Preprocess", duration, "nano"));
        }
    }

    private void forwardEnd(NDList list) {
        if (metrics != null) {
            // JnaUtils.waitAll();
            for (Pair<String, NDArray> pair : list) {
                ((MxNDArray) pair.getValue()).waitToRead();
            }
            long tmp = System.nanoTime();
            long duration = tmp - timestamp;
            timestamp = tmp;
            metrics.addMetric(new Metric("Inference", duration, "nano"));
        }
    }

    private void postProcessEnd() {
        if (metrics != null) {
            long tmp = System.nanoTime();
            long duration = tmp - timestamp;
            timestamp = tmp;
            metrics.addMetric(new Metric("Postprocess", duration, "nano"));
        }
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        cachedOp.close();
        factory.close();
    }

    private class PredictorContext implements TranslatorContext {

        private NDFactory ctxFactory;

        public PredictorContext() {
            ctxFactory = factory.newSubFactory();
        }

        /** {@inheritDoc} */
        @Override
        public Model getModel() {
            return model;
        }

        /** {@inheritDoc} */
        @Override
        public Context getContext() {
            return context;
        }

        /** {@inheritDoc} */
        @Override
        public NDFactory getNDFactory() {
            return ctxFactory;
        }

        /** {@inheritDoc} */
        @Override
        public Metrics getMetrics() {
            return metrics;
        }

        /** {@inheritDoc} */
        @Override
        public void close() {
            ctxFactory.close();
        }
    }
}
