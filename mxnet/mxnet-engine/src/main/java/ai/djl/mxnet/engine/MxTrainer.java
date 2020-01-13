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
package ai.djl.mxnet.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.training.LocalParameterServer;
import ai.djl.training.ParameterServer;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.TrainingListener;
import ai.djl.training.dataset.Batch;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.loss.Loss;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** {@code MxTrainer} is the MXNet implementation of the {@link Trainer}. */
public class MxTrainer implements Trainer {

    private static final Logger logger = LoggerFactory.getLogger(MxTrainer.class);

    private MxModel model;
    private MxNDManager manager;
    private Metrics metrics;
    private List<TrainingListener> listeners;
    private Device[] devices;
    private ParameterStore parameterStore;
    private List<Evaluator> trainingEvaluators;
    private List<Evaluator> validateEvaluators;
    private Loss trainingLoss;
    private Loss validationLoss;
    long batchBeginTime;

    private boolean gradientsChecked;

    /**
     * Creates an instance of {@code MxTrainer} with the given {@link MxModel} and {@link
     * TrainingConfig}.
     *
     * @param model the model the trainer will train on
     * @param trainingConfig the configuration used by the trainer
     */
    MxTrainer(MxModel model, TrainingConfig trainingConfig) {
        this.model = model;
        manager = (MxNDManager) model.getNDManager().newSubManager();
        devices = trainingConfig.getDevices();
        trainingLoss = trainingConfig.getLossFunction();
        if (trainingLoss == null) {
            throw new IllegalArgumentException("You must specify a loss for the trainer");
        }
        validationLoss = trainingLoss.duplicate();
        trainingEvaluators = new ArrayList<>(trainingConfig.getEvaluators());
        validateEvaluators = new ArrayList<>();
        trainingEvaluators.forEach(i -> validateEvaluators.add(i.duplicate()));

        // track loss as an evaluator by default
        trainingEvaluators.add(trainingLoss);
        // add from duplication of trainingLoss
        // do not mess up with duplication of evaluators
        validateEvaluators.add(validationLoss);

        // ParameterServer parameterServer = new MxParameterServer(trainingConfig.getOptimizer());
        ParameterServer parameterServer = new LocalParameterServer(trainingConfig.getOptimizer());

        parameterStore = new ParameterStore(manager, false);
        parameterStore.setParameterServer(parameterServer, devices);

        listeners = trainingConfig.getTrainingListeners();
        listeners.forEach(listener -> listener.onTrainingBegin(this));
    }

    /** {@inheritDoc} */
    @Override
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

    /** {@inheritDoc} */
    @Override
    public GradientCollector newGradientCollector() {
        return new MxGradientCollector();
    }

    /** {@inheritDoc} */
    @Override
    public void trainBatch(Batch batch) {
        if (manager.getEngine() != batch.getManager().getEngine()) {
            throw new IllegalArgumentException(
                    "The data must be on the same engine as the trainer. You may need to change one of your NDManagers.");
        }
        Batch[] splits = batch.split(devices, false);
        try (GradientCollector collector = new MxGradientCollector()) {
            for (Batch split : splits) {
                NDList data = split.getData();
                NDList labels = split.getLabels();
                NDList preds = forward(data);

                long time = System.nanoTime();
                NDArray loss = trainingLoss.getLoss(labels, preds);

                collector.backward(loss);
                addMetric("backward", time);
                time = System.nanoTime();

                updateEvaluators(labels, preds);
                addMetric("training-metrics", time);
            }
        }

        addMetric("train", batchBeginTime);
        // count batch begin time at end of batch to include batch loading time
        batchBeginTime = System.nanoTime();

        listeners.forEach(listener -> listener.onTrainingBatch(this));
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(NDList input) {
        long begin = System.nanoTime();
        try {
            return model.getBlock().forward(parameterStore, input);
        } finally {
            addMetric("forward", begin);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void validateBatch(Batch batch) {
        if (manager.getEngine() != batch.getManager().getEngine()) {
            throw new IllegalArgumentException(
                    "The data must be on the same engine as the trainer. You may need to change one of your NDManagers.");
        }
        long begin = System.nanoTime();
        Batch[] splits = batch.split(devices, false);
        for (Batch split : splits) {
            NDList data = split.getData();
            NDList labels = split.getLabels();

            NDList preds = forward(data);
            updateValidationMetrics(labels, preds);
        }
        addMetric("validate", begin);

        listeners.forEach(listener -> listener.onValidationBatch(this));
    }

    /** {@inheritDoc} */
    @Override
    public void step() {
        if (!gradientsChecked) {
            checkGradients();
        }

        long begin = System.nanoTime();
        parameterStore.updateAllParameters();
        addMetric("step", begin);
    }

    /** {@inheritDoc} */
    @Override
    public Metrics getMetrics() {
        return metrics;
    }

    /** {@inheritDoc} */
    @Override
    public void setMetrics(Metrics metrics) {
        this.metrics = metrics;
    }

    /** {@inheritDoc} */
    @Override
    public List<Device> getDevices() {
        return Arrays.asList(devices);
    }

    private void updateEvaluators(NDList labels, NDList preds) {
        // stop recording as this is end of computation graph
        // any evaluator calculation or update operation should not be recorded
        MxGradientCollector.setRecording(false);
        MxGradientCollector.setTraining(false);
        // this step is synchronized, should be done at end of batch
        trainingEvaluators.forEach(evaluator -> evaluator.update(labels, preds));
        // turn gradient recording back on
        MxGradientCollector.setRecording(true);
        MxGradientCollector.setTraining(true);
    }

    private void updateValidationMetrics(NDList labels, NDList preds) {
        validateEvaluators.forEach(evaluator -> evaluator.update(labels, preds));
    }

    /** {@inheritDoc} */
    @Override
    public void resetEvaluators() {
        trainingEvaluators.forEach(Evaluator::reset);
        validateEvaluators.forEach(Evaluator::reset);

        listeners.forEach(listener -> listener.onEpoch(this));
    }

    /** {@inheritDoc} */
    @Override
    public Loss getLoss() {
        return trainingLoss;
    }

    /** {@inheritDoc} */
    @Override
    public Loss getValidationLoss() {
        return validationLoss;
    }

    /** {@inheritDoc} */
    @Override
    public Model getModel() {
        return model;
    }

    /** {@inheritDoc} */
    @Override
    public List<Evaluator> getTrainingEvaluators() {
        return trainingEvaluators;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public final <T extends Evaluator> T getTrainingEvaluator(Class<T> clazz) {
        for (Evaluator evaluator : trainingEvaluators) {
            if (clazz.isInstance(evaluator)) {
                return (T) evaluator;
            }
        }
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public List<Evaluator> getValidationEvaluators() {
        return validateEvaluators;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public <T extends Evaluator> T getValidationEvaluator(Class<T> clazz) {
        for (Evaluator evaluator : validateEvaluators) {
            if (clazz.isInstance(evaluator)) {
                return (T) evaluator;
            }
        }
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getManager() {
        return manager;
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

    private void addMetric(String metricName, long begin) {
        if (metrics != null && begin > 0L) {
            metrics.addMetric(metricName, System.nanoTime() - begin);
        }
    }
}
