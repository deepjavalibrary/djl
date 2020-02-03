/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.fasttext.engine;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@code FtPredictor} is the fastText implementation of {@link Predictor}.
 *
 * @see Predictor
 */
public class FtPredictor<I, O> implements Predictor<I, O> {

    private FtModel model;
    private Translator<I, O> translator;

    /**
     * Constructs a {@code FtPredictor}.
     *
     * @param model the model to predict with
     * @param translator the translator to convert with input and output
     */
    FtPredictor(FtModel model, Translator<I, O> translator) {
        this.model = model;
        this.translator = translator;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("PMD.AvoidRethrowingException")
    public O predict(I input) throws TranslateException {
        try {
            TranslatorContext ctx = new FtPredictorContext(model, input);
            return translator.processOutput(ctx, null);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new TranslateException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public List<O> batchPredict(List<I> inputs) throws TranslateException {
        List<O> list = new ArrayList<>(inputs.size());
        for (I input : inputs) {
            list.add(predict(input));
        }
        return list;
    }

    /** {@inheritDoc} */
    @Override
    public void setMetrics(Metrics metrics) {}

    /** {@inheritDoc} */
    @Override
    public void close() {}

    private static class FtPredictorContext implements TranslatorContext {

        private Model model;
        private Map<String, Object> attachments;

        FtPredictorContext(Model model, Object input) {
            this.model = model;
            attachments = new ConcurrentHashMap<>();
            attachments.put("input", input);
        }

        /** {@inheritDoc} */
        @Override
        public Model getModel() {
            return model;
        }

        /** {@inheritDoc} */
        @Override
        public NDManager getNDManager() {
            return null;
        }

        /** {@inheritDoc} */
        @Override
        public Metrics getMetrics() {
            return null;
        }

        /** {@inheritDoc} */
        @Override
        public Object getAttachment(String key) {
            return attachments.get(key);
        }

        /** {@inheritDoc} */
        @Override
        public void setAttachment(String key, Object value) {
            attachments.put(key, value);
        }

        /** {@inheritDoc} */
        @Override
        public void close() {}
    }
}
