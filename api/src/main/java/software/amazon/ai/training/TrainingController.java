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

package software.amazon.ai.training;

import java.util.List;
import java.util.stream.Collectors;
import software.amazon.ai.Device;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.util.PairList;

public class TrainingController implements AutoCloseable {

    private PairList<String, Parameter> parameters;
    private Optimizer optimizer;
    private ParameterStore parameterStore;
    private boolean gradientsChecked;
    private boolean updateOnParameterStore;
    private Device[] devices;

    public TrainingController(
            PairList<String, Parameter> parameters, Optimizer optimizer, Device[] devices) {
        this(parameters, optimizer, devices, devices.length > 1, devices.length > 1);
    }

    public TrainingController(
            PairList<String, Parameter> parameters,
            Optimizer optimizer,
            Device[] devices,
            boolean updateOnParameterStore,
            boolean aggregateOnGPU) {
        this.parameters = parameters;
        this.optimizer = optimizer;
        this.devices = devices;
        this.updateOnParameterStore = updateOnParameterStore;
        parameterStore = Engine.getInstance().newParameterStore(optimizer, aggregateOnGPU);
    }

    /** Makes one step of parameter update. */
    public void step() {
        if (!gradientsChecked) {
            checkGradients();
        }
        if (updateOnParameterStore) {
            //            reduceGradientsOnParameterStore();
            //            updateOnParameterStore();
            fakeReduceOnParameterStore();
        } else {
            optimizer.updateAllParameters(parameters);
        }
    }

    /**
     * Check if all gradients are zeros, prevent users from calling step() without running {@code
     * backward}.
     */
    private void checkGradients() {
        List<NDArray> grads =
                parameters
                        .stream()
                        .map(pair -> pair.getValue().getArray().getGradient())
                        .collect(Collectors.toList());
        NDArray gradSum = NDArrays.stack(grads.stream().map(NDArray::sum).toArray(NDArray[]::new));
        float[] sums = gradSum.sum().toFloatArray();
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

    // TODO: remove this method, and make optimizer update methods protected
    void fakeReduceOnParameterStore() {
        int numDevices = devices.length;
        for (int i = 0; i < parameters.size(); i++) {
            NDArray[] grads = new NDArray[numDevices];
            NDArray[] paramArrays = new NDArray[numDevices];
            for (int j = 0; j < numDevices; j++) {
                paramArrays[j] = parameters.get(i).getValue().getArray(devices[j]);
                grads[j] = paramArrays[j].getGradient().asInDevice(Device.cpu(0), true);
            }
            NDArray stackedGrads = NDArrays.stack(grads);
            NDArray mean = stackedGrads.mean(new int[] {0});
            for (int j = 0; j < numDevices; j++) {
                NDArray meanGPU = mean.asInDevice(devices[j], true);
                optimizer.update(i, paramArrays[j], meanGPU);
                meanGPU.close();
            }
            mean.close();
            stackedGrads.close();
            for (int j = 0; j < numDevices; j++) {
                grads[j].close();
            }
        }
    }

    void reduceGradientsOnParameterStore() {
        for (int i = 0; i < parameters.size(); i++) {
            // TODO: handle gradient from multiple devices
            NDArray gradient = parameters.get(i).getValue().getArray().getGradient();
            parameterStore.push(i, gradient);
        }
    }

    void updateOnParameterStore() {
        for (int i = 0; i < parameters.size(); i++) {
            // TODO: handle update from multiple devices
            NDArray paramArray = parameters.get(i).getValue().getArray();
            parameterStore.pull(i, paramArray);
        }
    }

    @Override
    public void close() {
        parameterStore.close();
    }
}
