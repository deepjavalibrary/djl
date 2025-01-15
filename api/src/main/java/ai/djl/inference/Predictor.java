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

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.inference.streaming.StreamingBlock;
import ai.djl.inference.streaming.StreamingTranslator;
import ai.djl.inference.streaming.StreamingTranslator.StreamOutput;
import ai.djl.metric.Dimension;
import ai.djl.metric.Metrics;
import ai.djl.metric.Unit;
import ai.djl.ndarray.LazyNDArray;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The {@code Predictor} interface provides a session for model inference.
 *
 * <p>You can use a {@code Predictor}, with a specified {@link Translator}, to perform inference on
 * a {@link Model}. The following is example code that uses {@code Predictor}:
 *
 * <pre>
 * Model model = Model.load(modelDir, modelName);
 *
 * // User must implement Translator interface, read {@link Translator} for detail.
 * Translator&lt;String, String&gt; translator = new MyTranslator();
 *
 * try (Predictor&lt;String, String&gt; predictor = model.newPredictor(translator)) {
 *   String result = predictor.predict("What's up");
 * }
 * </pre>
 *
 * <p>See the tutorials on:
 *
 * <ul>
 *   <li><a
 *       href="https://docs.djl.ai/master/docs/demos/jupyter/tutorial/03_image_classification_with_your_model.html">Inference
 *       with a custom trained model</a>
 *   <li><a
 *       href="https://docs.djl.ai/master/docs/demos/jupyter/object_detection_with_model_zoo.html">Inference
 *       with a model zoo model</a>
 *   <li><a href="https://docs.djl.ai/master/docs/demos/jupyter/load_mxnet_model.html">Inference
 *       with an MXNet model</a>
 * </ul>
 *
 * @param <I> the input type
 * @param <O> the output type
 * @see Model
 * @see Translator
 * @see <a href="https://docs.djl.ai/master/docs/development/memory_management.html">The guide on
 *     memory management</a>
 * @see <a
 *     href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/multithread_inference.md">The
 *     guide on running multi-threaded inference</a>
 * @see <a
 *     href="https://docs.djl.ai/master/docs/development/inference_performance_optimization.html">The
 *     guide on inference performance optimization</a>
 */
public class Predictor<I, O> implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(Predictor.class);

    protected Translator<I, O> translator;
    protected long timestamp;

    protected boolean prepared;
    protected Model model;
    protected NDManager manager;
    protected Metrics metrics;
    protected Block block;
    protected ParameterStore parameterStore;
    protected Dimension dimension;

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
    public Predictor(Model model, Translator<I, O> translator, Device device, boolean copy) {
        if (!device.equals(model.getNDManager().getDevice())) {
            // Always copy during device changes
            copy = true;
        }
        this.model = model;
        this.manager = model.getNDManager().newSubManager(device);
        this.manager.setName("predictor");
        this.translator = translator;
        block = model.getBlock();
        parameterStore = new ParameterStore(manager, copy);
        dimension = new Dimension("Model", model.getProperty("metric_dimension", "model"));
    }

    /**
     * Predicts an item for inference.
     *
     * @param input the input
     * @return the output object defined by the user
     * @throws TranslateException if an error occurs during prediction
     */
    public O predict(I input) throws TranslateException {
        return batchPredict(Collections.singletonList(input)).get(0);
    }

    /**
     * Predicts an item for inference.
     *
     * @param ctx the context for the {@code Predictor}.
     * @param ndList the input {@code NDList}
     * @return the output {@code NDList}
     * @throws TranslateException if an error occurs during prediction
     */
    protected NDList predictInternal(TranslatorContext ctx, NDList ndList)
            throws TranslateException {
        logger.trace("Predictor input data: {}", ndList);
        // TODO: this check was (partially) introduced in
        // https://github.com/deepjavalibrary/djl/pull/3579,
        // but should ideally not be needed.
        if (ndList != null && ndList.isEmpty()) {
            return new NDList();
        }
        return block.forward(parameterStore, ndList, false);
    }

    /**
     * Predicts a batch for inference.
     *
     * @param inputs a list of inputs
     * @return a list of output objects defined by the user
     * @throws TranslateException if an error occurs during prediction
     */
    @SuppressWarnings({"PMD.AvoidRethrowingException", "PMD.IdenticalCatchBranches", "unchecked"})
    public List<O> batchPredict(List<I> inputs) throws TranslateException {
        try (PredictorContext context = new PredictorContext()) {
            if (!prepared) {
                translator.prepare(context);
                prepared = true;
            }
            if (translator.getBatchifier() == null) {
                List<O> ret = new ArrayList<>(inputs.size());
                for (I input : inputs) {
                    timestamp = System.nanoTime();
                    long begin = timestamp;
                    NDList ndList = translator.processInput(context, input);
                    preprocessEnd(ndList, 1);

                    NDList result = predictInternal(context, ndList);
                    predictEnd(result, 1);

                    ret.add(translator.processOutput(context, result));
                    postProcessEnd(begin, 1);
                }
                return ret;
            }

            int batchSize = inputs.size();

            timestamp = System.nanoTime();
            long begin = timestamp;
            NDList ndList = translator.batchProcessInput(context, inputs);
            preprocessEnd(ndList, batchSize);

            NDList result = predictInternal(context, ndList);
            predictEnd(result, batchSize);

            List<O> ret = translator.batchProcessOutput(context, result);
            postProcessEnd(begin, batchSize);
            return ret;
        } catch (TranslateException e) {
            throw e;
        } catch (Exception e) {
            throw new TranslateException(e);
        }
    }

    /**
     * Predicts an item for inference.
     *
     * @param input the input
     * @return the output object defined by the user
     * @throws TranslateException if an error occurs during prediction
     */
    @SuppressWarnings({"PMD.AvoidRethrowingException", "PMD.IdenticalCatchBranches"})
    public StreamOutput<O> streamingPredict(I input) throws TranslateException {

        String streamingSupported = streamingSupportError();
        if (streamingSupported != null) {
            throw new IllegalStateException(streamingSupported);
        }

        StreamingBlock streamingBlock = (StreamingBlock) block;
        StreamingTranslator<I, O> streamingTranslator = (StreamingTranslator<I, O>) translator;

        try {
            PredictorContext context = new PredictorContext();
            if (!prepared) {
                translator.prepare(context);
                prepared = true;
            }
            Batchifier batchifier = translator.getBatchifier();
            if (batchifier == null) {
                NDList ndList = translator.processInput(context, input);

                return streamingTranslator.processStreamOutput(
                        context,
                        streamingBlock
                                .forwardStream(parameterStore, ndList, false)
                                .onClose(context::close));
            }

            // For the batched case, need to create singleton batch and unbatchify singleton
            NDList inputBatch = processInputs(context, Collections.singletonList(input));
            return streamingTranslator.processStreamOutput(
                    context,
                    streamingBlock
                            .forwardStream(parameterStore, inputBatch, false)
                            .map(
                                    result -> {
                                        NDList[] unbatched =
                                                translator.getBatchifier().unbatchify(result);
                                        if (unbatched.length != 1) {
                                            throw new IllegalStateException(
                                                    "Unexpected number of outputs from model");
                                        }
                                        return unbatched[0];
                                    })
                            .onClose(context::close));

        } catch (TranslateException e) {
            throw e;
        } catch (Exception e) {
            throw new TranslateException(e);
        }
    }

    /**
     * Returns true if streaming is supported by the predictor, block, and translator.
     *
     * @return true if streaming is supported by the predictor, block, and translator
     */
    public boolean supportsStreaming() {
        return streamingSupportError() == null;
    }

    private String streamingSupportError() {
        if (!(block instanceof StreamingBlock)) {
            return "streamingPredict() can only be called with a StreamingBlock";
        }
        if (!(translator instanceof StreamingTranslator)) {
            return "streamingPredict() can only be called with a StreamingTranslator";
        }
        return null;
    }

    /**
     * Attaches a Metrics param to use for benchmark.
     *
     * @param metrics the Metrics class
     */
    public void setMetrics(Metrics metrics) {
        this.metrics = metrics;
    }

    private void waitToRead(NDList list) {
        for (NDArray array : list) {
            if (array instanceof LazyNDArray) {
                ((LazyNDArray) array).waitToRead();
            }
        }
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

    private void preprocessEnd(NDList list, int batchSize) {
        if (metrics != null) {
            waitToRead(list);
            long tmp = System.nanoTime();
            long duration = (tmp - timestamp) / 1000 / batchSize;
            timestamp = tmp;
            metrics.addMetric("Preprocess", duration, Unit.MICROSECONDS, dimension);
        }
    }

    private void predictEnd(NDList list, int batchSize) {
        if (metrics != null) {
            waitToRead(list);
            long tmp = System.nanoTime();
            long duration = (tmp - timestamp) / 1000 / batchSize;
            timestamp = tmp;
            metrics.addMetric("Inference", duration, Unit.MICROSECONDS, dimension);
        }
    }

    private void postProcessEnd(long begin, int batchSize) {
        if (metrics != null) {
            long tmp = System.nanoTime();
            long duration = (tmp - timestamp) / 1000 / batchSize;
            timestamp = tmp;
            metrics.addMetric("Postprocess", duration, Unit.MICROSECONDS, dimension);
            long prediction = (tmp - begin) / 1000;
            metrics.addMetric("Prediction", prediction, Unit.MICROSECONDS, dimension);
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
                logger.warn("Predictor for {} was not closed explicitly.", model.getName());
            }
            close();
        }
        super.finalize();
    }

    protected class PredictorContext implements TranslatorContext {

        private NDManager ctxManager;
        private Map<String, Object> attachments;

        /** Constructs a new {@code PredictorContext} instance. */
        public PredictorContext() {
            ctxManager = manager.newSubManager();
            ctxManager.setName("predictor ctx");
            attachments = new ConcurrentHashMap<>();
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
        public NDManager getPredictorManager() {
            return manager;
        }

        /** {@inheritDoc} */
        @Override
        public Block getBlock() {
            return block;
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
    }
}
