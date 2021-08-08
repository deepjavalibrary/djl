/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * a batch aggregator that terminates after a maximum idle time.
 *
 * @author erik.bamberg@web.de
 */
public class TemporaryBatchAggregator extends BatchAggregator {

    private static final Logger logger = LoggerFactory.getLogger(TemporaryBatchAggregator.class);

    private long idleSince;
    private long maxIdleTime;

    /**
     * a batch aggregator that terminates after a maximum idle time.
     *
     * @param model the model to run for.
     * @param jobQueue reference to external job queue for polling.
     */
    public TemporaryBatchAggregator(ModelInfo model, LinkedBlockingDeque<Job> jobQueue) {
        super(model, jobQueue);
        this.idleSince = System.currentTimeMillis();
        this.maxIdleTime = model.getMaxIdleTime();
    }

    /** {@inheritDoc} */
    @Override
    protected List<Job> pollBatch() throws InterruptedException {
        List<Job> list = new ArrayList<>(batchSize);
        Job job = jobQueue.poll(maxIdleTime, TimeUnit.SECONDS);
        if (job != null) {
            list.add(job);
            drainTo(list, maxBatchDelay);
            logger.trace("sending jobs, size: {}", list.size());
            idleSince = System.currentTimeMillis();
        }
        return list;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isFinished() {
        logger.trace(
                "check temporary batch aggregator idle time idle since {}ms - max idle time:{}ms",
                System.currentTimeMillis() - idleSince,
                maxIdleTime * 1000);
        return System.currentTimeMillis() - idleSince > maxIdleTime * 1000;
    }
}
