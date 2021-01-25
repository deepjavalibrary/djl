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
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.translate.TranslateException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class WorkerThread implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(WorkerThread.class);

    private ModelInfo model;
    private Predictor<Input, Output> predictor;

    private AtomicBoolean running = new AtomicBoolean(true);

    private BatchAggregator aggregator;
    private int gpuId;
    private AtomicReference<Thread> currentThread = new AtomicReference<>();
    private WorkerState state;
    private int workerId;
    private long startTime;
   

    private WorkerIDGenerator workerIDGenerator=new WorkerIDGenerator();
    
    public WorkerThread(int gpuId, ModelInfo model, BatchAggregator aggregator) {
        this.model = model;
        this.aggregator = aggregator;
        this.gpuId = gpuId;
        this.workerId = workerIDGenerator.generate();
        this.startTime = System.currentTimeMillis();
        predictor = model.getModel().newPredictor();
       
    }

    /** {@inheritDoc} */
    @Override
    public void run() {
        Thread thread = Thread.currentThread();
        thread.setName(getWorkerName());
        currentThread.set(thread);
        this.state=WorkerState.WORKER_STARTED;
        List<Input> req = null;
        try {
            while (isRunning()) {
        	List<Job> jobsToExecute=pollBatch();
        	if (jobsToExecute!=null && !jobsToExecute.isEmpty()) {
                    req = aggregator.getRequest(jobsToExecute);
                    try {
                        List<Output> reply = predictor.batchPredict(req);
                        aggregator.sendResponse(reply);
                    } catch (TranslateException e) {
                        logger.warn("Failed to predict", e);
                        aggregator.sendError();
                    }
        	}
                req = null;
            }
        } catch (InterruptedException e) {
            logger.debug("Shutting down the thread .. Scaling down.");
        } catch (Throwable t) {
            logger.error("Server error", t);
        } finally {
            currentThread.set(null);
            shutdown(WorkerState.WORKER_STOPPED);
            if (req != null) {
                aggregator.sendError();
            }
        }
    }
    
    /**
     * Fills in the list with a batch of jobs.
     *
     * @param list the batch queue to be filled
     * @throws InterruptedException if interrupted
     */
    private List<Job> pollBatch() throws InterruptedException {
     //   try {
     //   }
//            Job job = jobQueue.take();
//            logger.trace("get first job: {}", job.getRequestId());
//
//            list.add(job);
//            long begin = System.currentTimeMillis();
//            long maxDelay = model.getMaxBatchDelay();
//            for (int i = 0; i < model.getBatchSize() - 1 && maxDelay > 0; ++i) {
//                job = jobQueue.poll(maxDelay, TimeUnit.MILLISECONDS);
//                if (job == null) {
//                    break;
//                }
//                long end = System.currentTimeMillis();
//                maxDelay -= end - begin;
//                begin = end;
//                list.add(job);
//            }
	// jobQueue.poll
	List<Job> list=new ArrayList<>(model.getBatchSize());
	Job job=jobQueue.poll(2, TimeUnit.SECONDS);
	if (job!=null) {
	    list.add(job);
	    jobQueue.drainTo(list,model.getBatchSize()-1);
            logger.trace("sending jobs, size: {}", list.size());
	}
        return list;
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
    
    /**
     * append a job into the jobQueue.
     * @param job
     * @return true if sucessfully added, false of capacity of the current work-queue exceeded
     */
    public boolean addJob(Job job) {
	if ((!running.get()) || state!=WorkerState.WORKER_STARTED)
	    return false;
        return jobQueue.offer(job);
    }

    public void shutdown(WorkerState state) {
        running.set(false);
        setState(state);
        Thread thread = currentThread.getAndSet(null);
        if (thread != null) {
            thread.interrupt();
            aggregator.sendError();
        }
        predictor.close();
    }

    private String getWorkerName() {
        String modelName = model.getModelName();
        if (modelName.length() > 25) {
            modelName = modelName.substring(0, 25);
        }
        return "W-" + modelName + '-' + workerId;
    }

    void setState(WorkerState newState) {
        logger.debug("{} State change {} -> {}", getWorkerName(), state, newState);
        if (state != WorkerState.WORKER_SCALED_DOWN) {
            // Don't update the state if it was terminated on purpose.. Scaling in..
            this.state = newState;
        }
    }
}
