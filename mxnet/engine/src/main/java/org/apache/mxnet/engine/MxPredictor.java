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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.nn.Block;
import software.amazon.ai.translate.TranslateException;
import software.amazon.ai.translate.Translator;
import software.amazon.ai.translate.TranslatorContext;
import software.amazon.ai.util.Pair;

/**
 * {@code MxPredictor} is the MXNet implementation of {@link Predictor}.
 *
 * <p>MxPredictor contains all methods in the Predictor class and MXNet specific implementations.
 *
 * @param <I> Input Object
 * @param <O> Output Object
 */
public class MxPredictor<I, O> implements Predictor<I, O> {

    private static final Logger logger = LoggerFactory.getLogger(MxPredictor.class);

    MxModel model;
    private Translator<I, O> translator;
    Context context;
    private Block block;
    MxNDManager manager;
    Metrics metrics;
    private long timestamp;

    MxPredictor(MxModel model, Translator<I, O> translator, Context context) {
        this.manager = MxNDManager.getSystemManager().newSubManager(context);
        this.model = model;
        this.translator = translator;
        this.context = context;
        this.block = model.getBlock();
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("PMD.AvoidRethrowingException")
    public O predict(I input) throws TranslateException {
        timestamp = System.nanoTime();

        try (PredictorContext inputCtx = new PredictorContext();
                PredictorContext outputCtx = new PredictorContext()) {
            NDList ndList = translator.processInput(inputCtx, input);
            preprocessEnd();

            NDList result = forward(ndList);
            forwardEnd(result);

            return translator.processOutput(outputCtx, result);
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

    private NDList forward(NDList ndList) {
        return block.forward(ndList);
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
            // JnaUtils.waitAll();
            for (Pair<String, NDArray> pair : list) {
                ((MxNDArray) pair.getValue()).waitToRead();
            }
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

    /** {@inheritDoc} */
    @SuppressWarnings("deprecation")
    @Override
    protected void finalize() throws Throwable {
        if (manager.isOpen()) {
            if (logger.isDebugEnabled()) {
                logger.warn("Model was not closed explicitly: {}", getClass().getSimpleName());
            }
            close();
        }
        super.finalize();
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
        public Context getContext() {
            return context;
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
