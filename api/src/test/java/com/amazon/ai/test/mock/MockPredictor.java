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
package com.amazon.ai.test.mock;

import com.amazon.ai.Context;
import com.amazon.ai.Model;
import com.amazon.ai.TranslateException;
import com.amazon.ai.Translator;
import com.amazon.ai.TranslatorContext;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.metric.Metrics;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.NDList;

public class MockPredictor<I, O> implements Predictor<I, O> {

    Model model;
    Context context;
    private Translator<I, O> translator;
    Metrics metrics;

    public MockPredictor(Model model, Translator<I, O> translator, Context context) {
        this.model = model;
        this.translator = translator;
        this.context = context;
    }

    @Override
    public O predict(I input) throws TranslateException {
        if (metrics != null) {
            metrics.addMetric("Preprocess", 1, "nano");
            metrics.addMetric("Inference", 3, "nano");
            metrics.addMetric("Postprocess", 2, "nano");
        }

        PredictorContext ctx = new PredictorContext();
        NDList ndList = translator.processInput(ctx, input);
        return translator.processOutput(ctx, ndList);
    }

    @Override
    public void setMetrics(Metrics metrics) {
        this.metrics = metrics;
    }

    @Override
    public void close() {}

    private class PredictorContext implements TranslatorContext {

        private NDFactory ctxFactory;

        public PredictorContext() {
            ctxFactory = new MockNDFactory();
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
