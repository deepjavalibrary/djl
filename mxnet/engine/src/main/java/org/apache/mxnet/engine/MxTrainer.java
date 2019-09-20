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
import software.amazon.ai.Device;
import software.amazon.ai.Model;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.nn.Block;
import software.amazon.ai.training.ModelSaver;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.TrainingController;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.translate.TranslatorContext;

public class MxTrainer implements Trainer {

    private static final Logger logger = LoggerFactory.getLogger(MxTrainer.class);

    private MxModel model;
    private Device[] devices;
    private Block block;
    private MxNDManager manager;
    private Metrics metrics;
    private TrainingController trainingController;

    MxTrainer(MxModel model, Device device) {
        this(model, new Device[] {device});
    }

    MxTrainer(MxModel model, Optimizer optimizer, Device device) {
        this(model, device);
        trainingController =
                new TrainingController(block.getParameters(), optimizer, new Device[] {device});
    }

    MxTrainer(MxModel model, Optimizer optimizer, Device[] devices) {
        this(model, devices);
        trainingController = new TrainingController(block.getParameters(), optimizer, devices);
    }

    MxTrainer(MxModel model, Device[] devices) {
        this.model = model;
        this.manager = (MxNDManager) model.getNDManager().newSubManager();
        this.devices = devices;
        this.block = model.getBlock();
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
    public NDList forward(NDList intput) {
        return block.forward(intput);
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
        public Device getDevice() {
            return devices[0];
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
