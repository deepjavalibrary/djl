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
package ai.djl.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.listener.EpochTrainingListener;
import ai.djl.training.listener.EvaluatorTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.listener.TrainingListener.BatchData;
import ai.djl.training.loss.Loss;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The {@code Trainer} interface provides a session for model training.
 *
 * <p>{@code Trainer} provides an easy, and manageable interface for training. {@code Trainer} is
 * not thread-safe.
 *
 * <p>See the tutorials on:
 *
 * <ul>
 *   <li><a
 *       href="https://github.com/awslabs/djl/blob/master/jupyter/tutorial/train_your_first_model.ipynb">Training
 *       your first model</a>
 *   <li><a
 *       href="https://github.com/awslabs/djl/blob/master/jupyter/transfer_learning_on_cifar10.ipynb">Training
 *       using transfer learning</a>
 *   <li><a
 *       href="https://github.com/awslabs/djl/blob/master/jupyter/load_mxnet_model.ipynb">Inference
 *       with an MXNet model</a>
 * </ul>
 */
public class Trainer implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(Trainer.class);

    private Model model;
    private NDManager manager;
    private Metrics metrics;
    private List<TrainingListener> listeners;
    private Device[] devices;
    private ParameterStore parameterStore;
    private List<Evaluator> evaluators;
    private Loss loss;
    private DataManager dataManager;
    long batchBeginTime;

    private boolean gradientsChecked;

    /**
     * Creates an instance of {@code Trainer} with the given {@link Model} and {@link
     * TrainingConfig}.
     *
     * @param model the model the trainer will train on
     * @param trainingConfig the configuration used by the trainer
     */
    public Trainer(Model model, TrainingConfig trainingConfig) {
        this.model = model;
        manager = model.getNDManager().newSubManager();
        devices = trainingConfig.getDevices();
        loss = trainingConfig.getLossFunction();
        dataManager = trainingConfig.getDataManager();
        if (loss == null) {
            throw new IllegalArgumentException("You must specify a loss for the trainer");
        }
        evaluators = new ArrayList<>(trainingConfig.getEvaluators());
        evaluators.add(loss); // track loss as an evaluator by default

        // ParameterServer parameterServer = new MxParameterServer(trainingConfig.getOptimizer());
        ParameterServer parameterServer = new LocalParameterServer(trainingConfig.getOptimizer());

        parameterStore = new ParameterStore(manager, false);
        parameterStore.setParameterServer(parameterServer, devices);

        listeners = trainingConfig.getTrainingListeners();
        listeners.forEach(listener -> listener.onTrainingBegin(this));
    }

    /**
     * Initializes the {@link Model} that the {@code Trainer} is going to train.
     *
     * @param shapes an array of {@code Shape} of the inputs
     */
    public void initialize(Shape... shapes) {
        model.getBlock().initialize(model.getNDManager(), model.getDataType(), shapes);
        // call getValue on all params to initialize on all devices
        model.getBlock()
                .getParameters()
                .forEach(
                        pair -> {
                            for (Device device : devices) {
                                parameterStore.getValue(pair.getValue(), device);
                            }
                        });
    }

    /**
     * Fetches an iterator that can iterate through the given {@link Dataset}.
     *
     * @param dataset the dataset to iterate through
     * @return an {@link Iterable} of {@link Batch} that contains batches of data from the dataset
     */
    public Iterable<Batch> iterateDataset(Dataset dataset) {
        return dataset.getData(getManager());
    }

    /**
     * Returns a new instance of {@link GradientCollector}.
     *
     * @return a new instance of {@link GradientCollector}
     */
    public GradientCollector newGradientCollector() {
        return manager.getEngine().newGradientCollector();
    }

    /**
     * Trains the model with one iteration of the given {@link Batch} of data.
     *
     * @param batch a {@link Batch} that contains data, and its respective labels
     * @throws IllegalArgumentException if the batch engine does not match the trainer engine
     */
    public void trainBatch(Batch batch) {
        if (manager.getEngine() != batch.getManager().getEngine()) {
            throw new IllegalArgumentException(
                    "The data must be on the same engine as the trainer. You may need to change one of your NDManagers.");
        }
        Batch[] splits = batch.split(devices, false);
        BatchData batchData =
                new BatchData(batch, new ConcurrentHashMap<>(), new ConcurrentHashMap<>());
        try (GradientCollector collector = newGradientCollector()) {
            for (Batch split : splits) {
                NDList data = dataManager.getData(split);
                NDList labels = dataManager.getLabels(split);
                NDList preds = forward(data);
                long time = System.nanoTime();
                NDArray lossValue = loss.evaluate(labels, preds);
                collector.backward(lossValue);
                addMetric("backward", time);
                time = System.nanoTime();
                batchData.getLabels().put(labels.get(0).getDevice(), labels);
                batchData.getPredictions().put(preds.get(0).getDevice(), preds);
                addMetric("training-metrics", time);
            }
        }

        addMetric("train", batchBeginTime);
        // count batch begin time at end of batch to include batch loading time
        batchBeginTime = System.nanoTime();

        listeners.forEach(listener -> listener.onTrainingBatch(this, batchData));
    }

    /**
     * Applies the forward function of the model once on the given input {@link NDList}.
     *
     * @param input the input {@link NDList}
     * @return the output of the forward function
     */
    public NDList forward(NDList input) {
        long begin = System.nanoTime();
        try {
            return model.getBlock().forward(parameterStore, input);
        } finally {
            addMetric("forward", begin);
        }
    }

    /**
     * Applies the predict function of the model once on the given input {@link NDList}.
     *
     * @param input the input {@link NDList}
     * @return the output of the predict function
     */
    public NDList predict(NDList input) {
        return model.getBlock().predict(parameterStore, input);
    }

    /**
     * Validates the given batch of data.
     *
     * <p>During validation, the evaluators and losses are computed, but gradients aren't computed,
     * and parameters aren't updated.
     *
     * @param batch a {@link Batch} of data
     * @throws IllegalArgumentException if the batch engine does not match the trainer engine
     */
    public void validateBatch(Batch batch) {
        if (manager.getEngine() != batch.getManager().getEngine()) {
            throw new IllegalArgumentException(
                    "The data must be on the same engine as the trainer. You may need to change one of your NDManagers.");
        }
        long begin = System.nanoTime();
        Batch[] splits = batch.split(devices, false);
        BatchData batchData =
                new BatchData(batch, new ConcurrentHashMap<>(), new ConcurrentHashMap<>());
        for (Batch split : splits) {
            NDList data = dataManager.getData(split);
            NDList labels = dataManager.getLabels(split);

            NDList preds = forward(data);
            batchData.getLabels().put(labels.get(0).getDevice(), labels);
            batchData.getPredictions().put(preds.get(0).getDevice(), preds);
        }
        addMetric("validate", begin);

        listeners.forEach(listener -> listener.onValidationBatch(this, batchData));
    }

    /** Updates all of the parameters of the model once. */
    public void step() {
        if (!gradientsChecked) {
            checkGradients();
        }

        long begin = System.nanoTime();
        parameterStore.updateAllParameters();
        addMetric("step", begin);
    }

    /**
     * Returns the Metrics param used for benchmarking.
     *
     * @return the the Metrics param used for benchmarking
     */
    public Metrics getMetrics() {
        return metrics;
    }

    /**
     * Attaches a Metrics param to use for benchmarking.
     *
     * @param metrics the Metrics class
     */
    public void setMetrics(Metrics metrics) {
        this.metrics = metrics;
    }

    /**
     * Returns the devices used for training.
     *
     * @return the devices used for training
     */
    public List<Device> getDevices() {
        return Arrays.asList(devices);
    }

    /** Runs the end epoch actions. */
    public void endEpoch() {
        listeners.forEach(listener -> listener.onEpoch(this));
    }

    /**
     * Gets the training {@link Loss} function of the trainer.
     *
     * @return the {@link Loss} function
     */
    public Loss getLoss() {
        return loss;
    }

    /**
     * Returns the model used to create this trainer.
     *
     * @return the model associated with this trainer
     */
    public Model getModel() {
        return model;
    }

    /**
     * Gets all {@link Evaluator}s.
     *
     * @return the evaluators used during training
     */
    public List<Evaluator> getEvaluators() {
        return evaluators;
    }

    /**
     * Returns the {@link TrainingResult}.
     *
     * @return the {@code TrainingResult}
     */
    public TrainingResult getTrainingResult() {
        TrainingResult result = new TrainingResult();
        for (TrainingListener listener : listeners) {
            if (listener instanceof EpochTrainingListener) {
                result.setEpoch(((EpochTrainingListener) listener).getNumEpochs());
            } else if (listener instanceof EvaluatorTrainingListener) {
                EvaluatorTrainingListener l = (EvaluatorTrainingListener) listener;
                result.setEvaluations(l.getLatestEvaluations());
            }
        }
        return result;
    }

    /**
     * Gets the {@link NDManager} from the model.
     *
     * @return the {@link NDManager}
     */
    public NDManager getManager() {
        return manager;
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

    /** {@inheritDoc} */
    @Override
    public void close() {
        listeners.forEach(listener -> listener.onTrainingEnd(this));

        parameterStore.sync();
        manager.close();
    }

    /**
     * Checks if all gradients are zeros. This prevent users from calling step() without running
     * {@code backward}.
     */
    private void checkGradients() {
        List<NDArray> grads = new ArrayList<>();
        model.getBlock()
                .getParameters()
                .values()
                .stream()
                .filter(Parameter::requireGradient)
                .forEach(
                        param ->
                                grads.add(
                                        parameterStore.getValue(param, devices[0]).getGradient()));

        NDList list = new NDList(grads.stream().map(NDArray::sum).toArray(NDArray[]::new));
        NDArray gradSum = NDArrays.stack(list);
        list.close();

        NDArray array = gradSum.sum();

        float[] sums = array.toFloatArray();

        array.close();
        gradSum.close();

        float sum = 0f;
        for (float num : sums) {
            sum += num;
        }
        if (sum == 0f) {
            throw new IllegalStateException(
                    "Gradient values are all zeros, please call gradientCollector.backward() on"
                            + "your target NDArray (usually loss), before calling step() ");
        }

        gradientsChecked = true;
    }

    private void addMetric(String metricName, long begin) {
        if (metrics != null && begin > 0L) {
            metrics.addMetric(metricName, System.nanoTime() - begin);
        }
    }
}
