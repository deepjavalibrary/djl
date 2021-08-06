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
package ai.djl.serving.http;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/** A class that holds information about model status. */
public class DescribeModelResponse {

    private String modelName;
    private String modelUrl;
    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private int maxBatchDelay;
    private int maxIdleTime;
    private int queueLength;
    private String status;
    private boolean loadedAtStartup;

    private List<Worker> workers;

    /** Constructs a {@code DescribeModelResponse} instance. */
    public DescribeModelResponse() {
        workers = new ArrayList<>();
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
     * Sets the model name.
     *
     * @param modelName the model name
     */
    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    /**
     * Returns if the models was loaded at startup.
     *
     * @return {@code true} if the models was loaded at startup
     */
    public boolean isLoadedAtStartup() {
        return loadedAtStartup;
    }

    /**
     * Sets the load at startup status.
     *
     * @param loadedAtStartup {@code true} if the models was loaded at startup
     */
    public void setLoadedAtStartup(boolean loadedAtStartup) {
        this.loadedAtStartup = loadedAtStartup;
    }

    /**
     * Returns the model URL.
     *
     * @return the model URL
     */
    public String getModelUrl() {
        return modelUrl;
    }

    /**
     * Sets the model URL.
     *
     * @param modelUrl the model URL
     */
    public void setModelUrl(String modelUrl) {
        this.modelUrl = modelUrl;
    }

    /**
     * Returns the desired minimum number of workers.
     *
     * @return the desired minimum number of workers
     */
    public int getMinWorkers() {
        return minWorkers;
    }

    /**
     * Sets the desired minimum number of workers.
     *
     * @param minWorkers the desired minimum number of workers
     */
    public void setMinWorkers(int minWorkers) {
        this.minWorkers = minWorkers;
    }

    /**
     * Returns the desired maximum number of workers.
     *
     * @return the desired maximum number of workers
     */
    public int getMaxWorkers() {
        return maxWorkers;
    }

    /**
     * Sets the desired maximum number of workers.
     *
     * @param maxWorkers the desired maximum number of workers
     */
    public void setMaxWorkers(int maxWorkers) {
        this.maxWorkers = maxWorkers;
    }

    /**
     * Returns the batch size.
     *
     * @return the batch size
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
     * Returns the number of request in the queue.
     *
     * @return the number of request in the queue
     */
    public int getQueueLength() {
        return queueLength;
    }

    /**
     * Sets the number of request in the queue.
     *
     * @param queueLength the number of request in the queue
     */
    public void setQueueLength(int queueLength) {
        this.queueLength = queueLength;
    }

    /**
     * Returns the model's status.
     *
     * @return the model's status
     */
    public String getStatus() {
        return status;
    }

    /**
     * Sets the model's status.
     *
     * @param status the model's status
     */
    public void setStatus(String status) {
        this.status = status;
    }

    /**
     * Sets the max idle time for worker threads.
     *
     * @param maxIdleTime the time a worker thread can be idle before scaling down.
     */
    public void setMaxIdleTime(int maxIdleTime) {
        this.maxIdleTime = maxIdleTime;
    }

    /**
     * Returns the maximum idle time for worker threads.
     *
     * @return the maxIdleTime
     */
    public int getMaxIdleTime() {
        return maxIdleTime;
    }

    /**
     * Returns all workers information of the model.
     *
     * @return all workers information of the model
     */
    public List<Worker> getWorkers() {
        return workers;
    }

    /**
     * Adds worker to the worker list.
     *
     * @param id the worker's ID
     * @param startTime the worker's start time
     * @param isRunning {@code true} if worker is running
     * @param gpuId the GPU id assigned to the worker, -1 for CPU
     */
    public void addWorker(int id, long startTime, boolean isRunning, int gpuId) {
        Worker worker = new Worker();
        worker.setId(id);
        worker.setStartTime(new Date(startTime));
        worker.setStatus(isRunning ? "READY" : "UNLOADING");
        worker.setGpu(gpuId >= 0);
        workers.add(worker);
    }

    /** A class that holds workers information. */
    public static final class Worker {

        private int id;
        private Date startTime;
        private String status;
        private boolean gpu;

        /**
         * Returns the worker's ID.
         *
         * @return the worker's ID
         */
        public int getId() {
            return id;
        }

        /**
         * Sets the worker's ID.
         *
         * @param id the workers ID
         */
        public void setId(int id) {
            this.id = id;
        }

        /**
         * Returns the worker's start time.
         *
         * @return the worker's start time
         */
        public Date getStartTime() {
            return startTime;
        }

        /**
         * Sets the worker's start time.
         *
         * @param startTime the worker's start time
         */
        public void setStartTime(Date startTime) {
            this.startTime = startTime;
        }

        /**
         * Returns the worker's status.
         *
         * @return the worker's status
         */
        public String getStatus() {
            return status;
        }

        /**
         * Sets the worker's status.
         *
         * @param status the worker's status
         */
        public void setStatus(String status) {
            this.status = status;
        }

        /**
         * Return if the worker using GPU.
         *
         * @return {@code true} if the worker using GPU
         */
        public boolean isGpu() {
            return gpu;
        }

        /**
         * Sets if the worker using GPU.
         *
         * @param gpu if the worker using GPU
         */
        public void setGpu(boolean gpu) {
            this.gpu = gpu;
        }
    }
}
