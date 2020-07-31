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
package ai.djl.serving.wlm;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.ZooModel;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A class represent a loaded model and it's metadata. */
public class ModelInfo implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(ModelInfo.class);

    private String modelName;
    private String modelUrl;

    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private int maxBatchDelay;
    private ReentrantLock lock;

    private LinkedBlockingDeque<Job> jobs;

    private ZooModel<Input, Output> model;

    /**
     * Constructs a new {@code ModelInfo} instance.
     *
     * @param modelName the name of the model that will be used as HTTP endpoint
     * @param modelUrl the model url
     * @param model the {@link ZooModel}
     * @param queueSize the maximum request queue size
     */
    public ModelInfo(
            String modelName, String modelUrl, ZooModel<Input, Output> model, int queueSize) {
        this.modelName = modelName;
        this.modelUrl = modelUrl;
        this.model = model;
        batchSize = 1;
        maxBatchDelay = 100;
        jobs = new LinkedBlockingDeque<>(queueSize);
        lock = new ReentrantLock();
    }

    /**
     * Returns the loaded {@link ZooModel}.
     *
     * @return the loaded {@link ZooModel}
     */
    public ZooModel<Input, Output> getModel() {
        return model;
    }

    /**
     * Returns the model name.
     *
     * @return the model name
     */
    public String getModelName() {
        return modelName;
    }

    /**
     * Returns the model url.
     *
     * @return the model url
     */
    public String getModelUrl() {
        return modelUrl;
    }

    /**
     * Returns the model cache directory.
     *
     * @return the model cache directory
     */
    public Path getModelDir() {
        return model.getModelPath();
    }

    /**
     * Returns the configured minimum number of workers.
     *
     * @return the configured minimum number of workers
     */
    public int getMinWorkers() {
        return minWorkers;
    }

    /**
     * Sets the minimum number of workers.
     *
     * @param minWorkers the minimum number of workers
     */
    public void setMinWorkers(int minWorkers) {
        this.minWorkers = minWorkers;
    }

    /**
     * Returns the configured maximum number of workers.
     *
     * @return the configured maximum number of workers
     */
    public int getMaxWorkers() {
        return maxWorkers;
    }

    /**
     * Sets the maximum number of workers.
     *
     * @param maxWorkers the maximum number of workers
     */
    public void setMaxWorkers(int maxWorkers) {
        this.maxWorkers = maxWorkers;
    }

    /**
     * Returns the configured batch size.
     *
     * @return the configured batch size
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Sets the batch size.
     *
     * @param batchSize the batch size
     */
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    /**
     * Returns the maximum delay in milliseconds to aggregate a batch.
     *
     * @return the maximum delay in milliseconds to aggregate a batch
     */
    public int getMaxBatchDelay() {
        return maxBatchDelay;
    }

    /**
     * Sets the maximum delay in milliseconds to aggregate a batch.
     *
     * @param maxBatchDelay the maximum delay in milliseconds to aggregate a batch
     */
    public void setMaxBatchDelay(int maxBatchDelay) {
        this.maxBatchDelay = maxBatchDelay;
    }

    /**
     * Adds a job to the queue.
     *
     * @param job an inference job
     * @return {@code true} if the queue is full
     */
    public boolean addJob(Job job) {
        return jobs.offer(job);
    }

    /**
     * Fills in the list with a batch of jobs.
     *
     * @param list the batch queue to be filled
     * @throws InterruptedException if interrupted
     */
    public void pollBatch(List<Job> list) throws InterruptedException {
        try {
            lock.lockInterruptibly();
            Job job = jobs.take();
            logger.trace("get first job: {}", job.getRequestId());

            list.add(job);
            long begin = System.currentTimeMillis();
            long maxDelay = maxBatchDelay;
            for (int i = 0; i < batchSize - 1 && maxDelay > 0; ++i) {
                job = jobs.poll(maxDelay, TimeUnit.MILLISECONDS);
                if (job == null) {
                    break;
                }
                long end = System.currentTimeMillis();
                maxDelay -= end - begin;
                begin = end;
                list.add(job);
            }
            logger.trace("sending jobs, size: {}", list.size());
        } finally {
            lock.unlock();
        }
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (model != null) {
            model.close();
        }
    }
}
