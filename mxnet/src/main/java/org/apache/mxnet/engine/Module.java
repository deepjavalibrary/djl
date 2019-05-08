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

import com.amazon.ai.Context;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.GradReq;
import com.amazon.ai.util.ResourceAllocator;
import java.util.Collections;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Module implements AutoCloseable {

    static final Logger logger = LoggerFactory.getLogger(Module.class);

    private MxExecutor[] executors;
    private boolean forTraining;

    Module(MxExecutor[] executors, boolean forTraining) {
        this.executors = executors;
        this.forTraining = forTraining;
    }

    public NDList forward(NDList ndList) {
        for (MxExecutor executor : executors) {
            executor.forward(ndList, forTraining);
        }
        return getOutputs();
    }

    public NDList getOutputs() {
        if (executors.length == 1) {
            return new CloseShieldNDList(executors[0].getOutputs());
        }

        NDList ret = new CloseShieldNDList();
        for (MxExecutor executor : executors) {
            NDList out = executor.getOutputs();
            ret.addAll(out);
        }
        return ret;
    }

    public static Builder forTraining(Context context, MxModel model) {
        return new Builder(context, model, true);
    }

    public static Builder forInference(Context context, MxModel model) {
        return new Builder(context, model, true);
    }

    @Override
    public void close() {
        for (MxExecutor executor : executors) {
            executor.close();
        }
    }

    public static final class Builder {

        private Context context;
        private MxModel model;
        private boolean forTraining;
        private GradReq gradReq;

        private Map<String, Context> contextMap;

        public Builder(Context context, MxModel model, boolean forTraining) {
            this.context = context;
            this.model = model;
            this.forTraining = forTraining;

            gradReq = forTraining ? GradReq.WRITE : GradReq.NULL;
            contextMap = Collections.emptyMap();
        }

        public void setContextMap(Map<String, Context> contextMap) {
            this.contextMap = contextMap;
        }

        public void setGradReq(GradReq gradReq) {
            this.gradReq = gradReq;
        }

        public Module build(ResourceAllocator alloc) {
            if (forTraining) {
                if (gradReq == null) {
                    gradReq = GradReq.WRITE;
                }
            }

            Symbol symbol = model.getSymbol();
            String[] labelNames = model.getLabelNames();
            String[] stateNames = model.getOptimizerStates();
            MxExecutor[] executors =
                    symbol.simpleBind(
                            model,
                            Collections.singletonList(context),
                            labelNames,
                            stateNames,
                            gradReq,
                            contextMap,
                            null);

            if (logger.isDebugEnabled()) {
                long bytesExec = 0;
                for (MxExecutor executor : executors) {
                    bytesExec += executor.getExecutedBytes();
                }
                logger.debug("total executor bytes: {}.", bytesExec);
            }

            return new Module(executors, forTraining);
        }
    }

    private static final class CloseShieldNDList extends NDList {

        public CloseShieldNDList() {}

        public CloseShieldNDList(NDList other) {
            super(other);
        }

        @Override
        public void close() {}
    }
}
