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

import com.amazon.ai.util.Utils;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.apache.mxnet.Context;
import org.apache.mxnet.types.GradReq;
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

    public void forward(NdArray[] ndArrays) {
        for (MxExecutor executor : executors) {
            executor.forward(ndArrays, forTraining);
        }
    }

    public NdArray[] getOutputs() {
        if (executors.length == 1) {
            return executors[0].getOutputs();
        }

        int count = 0;
        for (MxExecutor executor : executors) {
            NdArray[] out = executor.getOutputs();
            count += out.length;
        }

        NdArray[] ret = new NdArray[count];
        int index = 0;
        for (MxExecutor executor : executors) {
            NdArray[] out = executor.getOutputs();
            System.arraycopy(out, 0, ret, index, out.length);
            index += out.length;
        }
        return ret;
    }

    public static Builder forTraining(Context context, MxModel model, List<DataDesc> descriptors) {
        return new Builder(context, model, descriptors, true);
    }

    public static Builder forInference(Context context, MxModel model, List<DataDesc> descriptors) {
        return new Builder(context, model, descriptors, true);
    }

    @Override
    public void close() {
        for (MxExecutor executor : executors) {
            executor.close();
        }
    }

    public static final class Builder {

        private List<Context> contexts;
        private MxModel model;
        private List<DataDesc> descriptors;
        private boolean forTraining;
        private GradReq gradReq;

        private String[] fixedParameters;
        private Map<String, Context> contextMap;

        public Builder(
                Context context, MxModel model, List<DataDesc> descriptors, boolean forTraining) {
            this.contexts = Collections.singletonList(context);
            this.model = model;
            this.descriptors = descriptors;
            this.forTraining = forTraining;

            gradReq = forTraining ? GradReq.WRITE : GradReq.NULL;
            contextMap = Collections.emptyMap();
        }

        public void setFixedParameters(String[] fixedParameters) {
            this.fixedParameters = fixedParameters;
        }

        public void setContextMap(Map<String, Context> contextMap) {
            this.contextMap = contextMap;
        }

        public void setGradReq(GradReq gradReq) {
            this.gradReq = gradReq;
        }

        public Module build(ResourceAllocator alloc) {
            String[] dataNames = new String[descriptors.size()];
            for (int i = 0; i < descriptors.size(); ++i) {
                dataNames[i] = descriptors.get(i).getName();
            }
            validate(dataNames, "data", true);
            validate(model.getLabelNames(), "label", false);
            if (forTraining) {
                validate(model.getOptimizerStates(), "state", true);
                validate(fixedParameters, "fixed_param", true);
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
                            contexts,
                            descriptors,
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

        private void validate(String[] names, String typeName, boolean required) {
            if (names == null || names.length == 0) {
                return;
            }

            String[] args = model.getSymbol().getArgParams();
            for (String name : names) {
                if (!Utils.contains(args, name)) {
                    String msg =
                            String.format(
                                    "Input %s_%s is not found in symbol.list_arguments().",
                                    typeName, name);
                    if (required) {
                        throw new IllegalArgumentException(msg);
                    }
                    logger.warn(msg);
                }
            }
        }
    }
}
