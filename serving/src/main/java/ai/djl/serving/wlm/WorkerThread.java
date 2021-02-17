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

import ai.djl.inference.Predictor;
import ai.djl.translate.TranslateException;
import java.util.List;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

final class WorkerThread implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(WorkerThread.class);

    private String workerName;
    private Predictor<Object, Object> predictor;

    private AtomicBoolean running = new AtomicBoolean(true);

    private BatchAggregator aggregator;
    private int gpuId;
    private AtomicReference<Thread> currentThread = new AtomicReference<>();
    private WorkerState state;
    private int workerId;
    private long startTime;
    private boolean fixPoolThread;

    /**
     * Builds a workerThread with this builder.
     *
     * @param builder build a new worker thread using this builder.
     */
    @SuppressWarnings("unchecked")
    private WorkerThread(Builder builder) {
        this.workerName = buildWorkerName(builder.model);
        this.aggregator = builder.aggregator;
        this.gpuId = builder.gpuId;
        this.workerId = new WorkerIdGenerator().generate();
        this.startTime = System.currentTimeMillis();
        predictor = (Predictor<Object, Object>) builder.model.getModel().newPredictor();
        this.fixPoolThread = builder.fixPoolThread;
    }

    /** {@inheritDoc} */
    @Override
    public void run() {
        Thread thread = Thread.currentThread();
        thread.setName(workerName);
        currentThread.set(thread);
        this.state = WorkerState.WORKER_STARTED;
        List<Job> jobs = null;
        try {
            while (isRunning() && !aggregator.isFinished()) {
                jobs = aggregator.getRequest();
                if (jobs != null && !jobs.isEmpty()) {
                    try {
                        List<Object> inputs =
                                jobs.stream().map(j -> j.getInput()).collect(Collectors.toList());
                        List<Object> reply = predictor.batchPredict(inputs);
                        sendResponse(jobs, reply);
                    } catch (TranslateException e) {
                        logger.warn("Failed to predict", e);
                        aggregator.sendError(jobs);
                    }
                }
                jobs = null;
            }

        } catch (InterruptedException e) {
            logger.debug("Shutting down the thread .. Scaling down.");
        } catch (Throwable t) {
            logger.error("Server error", t);
        } finally {
            logger.debug("Shutting down worker thread .. {}", currentThread.get().getName());
            currentThread.set(null);
            shutdown(WorkerState.WORKER_STOPPED);
            if (jobs != null) {
                aggregator.sendError(jobs);
            }
        }
    }

    /**
     * Sends to response to all waiting clients. the order of jobs and outputs have to be the same
     *
     * @param jobs list of client-jobs to send a repsone to
     * @param outputs list of model-outputs in same order as the input objects.
     */
    public void sendResponse(List<Job> jobs, List<Object> outputs) {
        if (jobs.size() != outputs.size()) {
            throw new IllegalStateException("Not all jobs get response.");
        }

        int i = 0;
        for (Object output : outputs) {
            jobs.get(i++).sendOutput(output);
        }
    }

    public int getWorkerId() {
        return workerId;
    }

    public boolean isRunning() {
        return running.get();
    }

    public int getGpuId() {
        return gpuId;
    }

    public long getStartTime() {
        return startTime;
    }

    public WorkerState getState() {
        return state;
    }

    public void shutdown(WorkerState state) {
        running.set(false);
        setState(state);
        Thread thread = currentThread.getAndSet(null);
        if (thread != null) {
            thread.interrupt();
        }
        predictor.close();
    }

    private String buildWorkerName(ModelInfo model) {
        String modelName = model.getModelName();
        if (modelName.length() > 25) {
            modelName = modelName.substring(0, 25);
        }
        return "W-" + modelName + '-' + workerId;
    }

    void setState(WorkerState newState) {
        logger.debug("{} State change {} -> {}", workerName, state, newState);
        if (state != WorkerState.WORKER_SCALED_DOWN) {
            // Don't update the state if it was terminated on purpose.. Scaling in..
            this.state = newState;
        }
    }

    /**
     * check if this worker is instantiate is one of the fix threads of a pool. fix threads are not
     * automatically scales down, so they are candidate for down scaling when minWorker/maxWorker
     * size of a model changes.
     *
     * @return the fixPoolThread
     */
    public boolean isFixPoolThread() {
        return fixPoolThread;
    }

    /**
     * Creates a builder to build a {@code WorkerThread}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** A Builder to construct a {@code WorkerThread}. */
    public static class Builder {

        private ModelInfo model;
        private BatchAggregator aggregator;
        private LinkedBlockingDeque<Job> jobQueue;
        private int gpuId;
        private boolean fixPoolThread;
        private GpuAssignmentStrategy gpuAssignmentStrategy;

        Builder() {
            this.gpuId = -1;
            this.fixPoolThread = true;
        }

        /**
         * Returns self reference to this builder.
         *
         * @return self reference to this builder
         */
        protected Builder self() {
            return this;
        }

        protected void preBuildProcessing() {
            if (aggregator == null) {
                if (fixPoolThread) {
                    aggregator = new PermanentBatchAggregator(model, jobQueue);
                } else {
                    aggregator = new TemporaryBatchAggregator(model, jobQueue);
                }
            }
            if (gpuAssignmentStrategy != null) {
                gpuId = gpuAssignmentStrategy.nextGpuId();
            }
        }

        protected void validate() {
            if (model == null) {
                throw new IllegalArgumentException("model must not be null");
            }
            if (jobQueue == null && aggregator == null) {
                throw new IllegalArgumentException(
                        "one of jobQueue or BatchAggregator have to be set.");
            }
        }

        /**
         * Builds the {@link WorkerThread} with the provided data.
         *
         * @return an {@link WorkerThread}
         */
        public WorkerThread build() {
            validate();
            preBuildProcessing();
            return new WorkerThread(this);
        }

        /**
         * Sets the {@code ModelInfo} the thread will be responsible for.
         *
         * @param model the model to set
         * @return self-reference to this builder.
         */
        public Builder setModel(ModelInfo model) {
            this.model = model;
            return self();
        }

        /**
         * Sets a {@code BatchAggregator} which overrides the instantiated default {@code
         * BatchAggregator}.
         *
         * @param aggregator the {@code BatchAggregator} to set
         * @return self-reference to this builder.
         */
        public Builder optAggregator(BatchAggregator aggregator) {
            this.aggregator = aggregator;
            return self();
        }

        /**
         * Sets the jobQueue used to poll for new jobs. The jobQueue is passed to the created
         * standard BatchAggregators if the Batch-Aggregator is not override using {@link
         * #optAggregator(BatchAggregator) optAggregator(BatchAggregator)}
         *
         * @param jobQueue the jobQueue to set
         * @return self-reference to this builder.
         */
        public Builder setJobQueue(LinkedBlockingDeque<Job> jobQueue) {
            this.jobQueue = jobQueue;
            return self();
        }

        /**
         * Sets the GPU ID for this worker thread. GPU ID = -1 for non GPU.
         *
         * <p>only used when {@link #optGpuAssignmentStrategy(GpuAssignmentStrategy)
         * optGpuAssignmentStrategy} is not set
         *
         * @param gpuId the gpuId to set. defaults to -1=no gpu
         * @return self-reference to this builder.
         */
        public Builder optGpuId(int gpuId) {
            this.gpuId = gpuId;
            return self();
        }

        /**
         * Sets if the workerThread should be part of the fixed pool. Fixed Pool workers don't
         * terminate themself but are managed by WorkLoadManager min/max-worker scale functionality.
         *
         * @param fixPoolThread the fixPoolThread to set
         * @return self-reference to this builder.
         */
        public Builder optFixPoolThread(boolean fixPoolThread) {
            this.fixPoolThread = fixPoolThread;
            return self();
        }

        /**
         * sets an optional strategy to assign gpuId to this workerThread. doesn't use any gpu
         * (gpuId=-1) when no {@code GpuAssignmentStrategy} is set.
         *
         * @param gpuAssignmentStrategy the gpuAssignmentStrategy to set
         * @return self-reference to this builder.
         */
        public Builder optGpuAssignmentStrategy(GpuAssignmentStrategy gpuAssignmentStrategy) {
            this.gpuAssignmentStrategy = gpuAssignmentStrategy;
            return self();
        }
    }
}
