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

import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.nn.Block;
import software.amazon.ai.training.ModelSaver;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.TrainingController;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.translate.TrainTranslator;
import software.amazon.ai.translate.TranslateException;
import software.amazon.ai.translate.TranslatorContext;
import software.amazon.ai.util.Pair;

public class MxTrainer<I, L, O> implements Trainer<I, L, O> {

    private static final Logger logger = LoggerFactory.getLogger(MxTrainer.class);

    private MxModel model;
    private TrainTranslator<I, L, O> translator;
    private Context context;
    private Block block;
    private MxNDManager manager;
    private Metrics metrics;
    private Integer seed;
    private long timestamp;
    private TrainingController trainingController;

    MxTrainer(MxModel model, TrainTranslator<I, L, O> translator, Context context) {
        this.model = model;
        this.manager = (MxNDManager) model.getNDManager().newSubManager();
        this.translator = translator;
        this.context = context;
        this.block = model.getBlock();
    }

    MxTrainer(
            MxModel model,
            TrainTranslator<I, L, O> translator,
            Optimizer optimizer,
            Context context) {
        this(model, translator, context);
        trainingController = new TrainingController(block.getParameters(), optimizer);
    }

    @Override
    public void step() {
        if (trainingController == null) {
            throw new IllegalStateException(
                    "No optimizer is set for trainer, please initialize"
                            + "your trainer with an Optimizer.");
        }
        trainingController.step();
    }

    @Override
    public TrainTranslator<I, L, O> getTranslator() {
        return translator;
    }

    @Override
    public TranslatorContext getPreprocessContext() {
        return new TrainerContext();
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("PMD.AvoidRethrowingException")
    public O predict(I input) throws TranslateException {
        timestamp = System.nanoTime();

        try (TrainerContext inputCtx = new TrainerContext();
                TrainerContext outputCtx = new TrainerContext()) {
            NDList ndList = translator.processInput(inputCtx, input);
            predictPreprocessEnd();

            NDList result = block.forward(ndList);
            predictForwardEnd(result);
            return translator.processOutput(outputCtx, result);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new TranslateException(e);
        } finally {
            predictPostProcessEnd();
        }
    }

    private void predictPreprocessEnd() {
        if (metrics != null) {
            long tmp = System.nanoTime();
            long duration = tmp - timestamp;
            timestamp = tmp;
            metrics.addMetric("PredictPreprocess", duration, "nano");
        }
    }

    private void predictForwardEnd(NDList list) {
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

    private void predictPostProcessEnd() {
        if (metrics != null) {
            long tmp = System.nanoTime();
            long duration = tmp - timestamp;
            timestamp = tmp;
            metrics.addMetric("PredictPostprocess", duration, "nano");
        }
    }

    @Override
    public void setMetrics(Metrics metrics) {
        this.metrics = metrics;
    }

    @Override
    public NDManager getManager() {
        return manager;
    }

    @Override
    public Optional<Integer> getSeed() {
        return Optional.ofNullable(seed);
    }

    @Override
    public void setSeed(int seed) {
        this.seed = seed;
    }

    @Override
    public ModelSaver getModelSaver() {
        return null;
    }

    @Override
    public void setModelSaver(ModelSaver modelSaver) {}

    @Override
    public void checkpoint() {}

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

    @Override
    public void close() {
        manager.close();
        if (trainingController != null) {
            trainingController.close();
        }
    }

    private class TrainerContext implements TranslatorContext {

        private NDManager ctxManager;

        TrainerContext() {
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
