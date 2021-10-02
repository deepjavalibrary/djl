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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
 *       href="https://github.com/deepjavalibrary/djl/blob/master/jupyter/tutorial/03_image_classification_with_your_model.ipynb">Inference
 *       with a custom trained model</a>
 *   <li><a
 *       href="https://github.com/deepjavalibrary/djl/blob/master/jupyter/object_detection_with_model_zoo.ipynb">Inference
 *       with a model zoo model</a>
 *   <li><a
 *       href="https://github.com/deepjavalibrary/djl/blob/master/jupyter/load_mxnet_model.ipynb">Inference
 *       with an MXNet model</a>
 * </ul>
 *
 * @param <I> the input type
 * @param <O> the output type
 * @see Model
 * @see Translator
 * @see <a href="http://docs.djl.ai/docs/development/memory_management.html">The guide on memory
 *     management</a>
 * @see <a
 *     href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/multithread_inference.md">The
 *     guide on running multi-threaded inference</a>
 * @see <a href="http://docs.djl.ai/docs/development/inference_performance_optimization.html">The
 *     guide on inference performance optimization</a>
 */
public class Predictor<I, O> implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(Predictor.class);
    private Translator<I, O> translator;
    private long timestamp;

    private boolean prepared;
    private Model model;
    protected NDManager manager;
    protected Metrics metrics;
    protected Block block;
    protected ParameterStore parameterStore;

    /**
     * Creates a new instance of {@code BasePredictor} with the given {@link Model} and {@link
     * Translator}.
     *
     * @param model the model on which the predictions are based
     * @param translator the translator to be used
     * @param copy whether to copy the parameters to the parameter store
     */
    public Predictor(Model model, Translator<I, O> translator, boolean copy) {
        this.model = model;
        this.manager = model.getNDManager().newSubManager();
        this.manager.setName("predictor");
        this.translator = translator;
        block = model.getBlock();
        parameterStore = new ParameterStore(manager, copy);
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
        return block.forward(parameterStore, ndList, false);
    }

    /**
     * Predicts a batch for inference.
     *
     * @param inputs a list of inputs
     * @return a list of output objects defined by the user
     * @throws TranslateException if an error occurs during prediction
     */
    @SuppressWarnings({"PMD.AvoidRethrowingException", "PMD.IdenticalCatchBranches"})
    public List<O> batchPredict(List<I> inputs) throws TranslateException {
        long begin = System.nanoTime();
        try (PredictorContext context = new PredictorContext()) {
            if (!prepared) {
                translator.prepare(context);
                prepared = true;
            }
            Batchifier batchifier = translator.getBatchifier();
            if (batchifier == null) {
                List<O> ret = new ArrayList<>(inputs.size());
                for (I input : inputs) {
                    timestamp = System.nanoTime();
                    begin = timestamp;
                    NDList ndList = translator.processInput(context, input);
                    preprocessEnd(ndList);

                    NDList result = predictInternal(context, ndList);
                    predictEnd(result);

                    ret.add(translator.processOutput(context, result));
                    postProcessEnd(begin);
                }
                return ret;
            }

            timestamp = System.nanoTime();
            NDList inputBatch = processInputs(context, inputs);
            preprocessEnd(inputBatch);

            NDList result = predictInternal(context, inputBatch);
            predictEnd(result);

            List<O> ret = processOutputs(context, result);
            postProcessEnd(begin);
            return ret;
        } catch (TranslateException e) {
            throw e;
        } catch (Exception e) {
            throw new TranslateException(e);
        }
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

    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    private List<O> processOutputs(TranslatorContext ctx, NDList list) throws Exception {
        NDList[] unbatched = translator.getBatchifier().unbatchify(list);
        List<O> outputs = new ArrayList<>(unbatched.length);
        for (NDList output : unbatched) {
            outputs.add(translator.processOutput(ctx, output));
        }
        return outputs;
    }

    private void preprocessEnd(NDList list) {
        if (metrics != null) {
            waitToRead(list);
            long tmp = System.nanoTime();
            long duration = tmp - timestamp;
            timestamp = tmp;
            metrics.addMetric("Preprocess", duration, "nano");
        }
    }

    private void predictEnd(NDList list) {
        if (metrics != null) {
            waitToRead(list);
            long tmp = System.nanoTime();
            long duration = tmp - timestamp;
            timestamp = tmp;
            metrics.addMetric("Inference", duration, "nano");
        }
    }

    private void postProcessEnd(long begin) {
        if (metrics != null) {
            long tmp = System.nanoTime();
            long duration = tmp - timestamp;
            timestamp = tmp;
            metrics.addMetric("Postprocess", duration, "nano");
            metrics.addMetric("Total", tmp - begin, "nano");
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

    private class PredictorContext implements TranslatorContext {

        private NDManager ctxManager;
        private Map<String, Object> attachments;

        PredictorContext() {
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
