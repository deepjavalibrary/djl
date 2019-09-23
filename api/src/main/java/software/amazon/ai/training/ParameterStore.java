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

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import software.amazon.ai.Device;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.PairList;

public class ParameterStore implements AutoCloseable {

    private PairList<String, Parameter> parameters;
    private ParameterServer parameterServer;
    private Map<Pair<Integer, Parameter>, NDArray> parameterValues = new ConcurrentHashMap<>();
    private Device[] devices;

    public ParameterStore(PairList<String, Parameter> parameters, Device[] devices) {
        this.parameters = parameters;
        this.devices = devices;
    }

    public void setParameterServer(ParameterServer parameterServer) {
        this.parameterServer = parameterServer;
    }

    public void updateAllParameters(Optimizer optimizer) {
        fakeReduceOnParameterStore(optimizer);
    }

    // TODO: remove this method, and make optimizer update methods protected
    void fakeReduceOnParameterStore(Optimizer optimizer) {
        int numDevices = devices.length;
        for (int i = 0; i < parameters.size(); i++) {
            NDArray[] grads = getAllGradients(parameters.get(i).getValue());
            NDArray[] paramValues = getAllValues(parameters.get(i).getValue());
            for (int j = 0; j < numDevices; j++) {
                grads[j] = grads[j].asInDevice(Device.cpu(0), true);
            }
            NDArray stackedGrads = NDArrays.stack(grads);
            NDArray mean = stackedGrads.mean(new int[] {0});
            for (int j = 0; j < numDevices; j++) {
                NDArray meanGPU = mean.asInDevice(devices[j], true);
                optimizer.update(i, paramValues[j], meanGPU);
                meanGPU.close();
            }
            mean.close();
            stackedGrads.close();
            for (int j = 0; j < numDevices; j++) {
                grads[j].close();
            }
        }
    }

    public void initialize(Parameter parameter, NDArray array) {
        for (Device device : devices) {
            NDArray arrayCopy = array.asInDevice(device, true);
            arrayCopy.attachGradient();
            parameterValues.put(new Pair<>(device.getDeviceId(), parameter), arrayCopy);
        }
    }

    public NDArray getValue(Parameter parameter, Device device) {
        NDArray value = parameterValues.get(new Pair<>(device.getDeviceId(), parameter));
        if (value == null) {
            throw new IllegalStateException(
                    "The parameter has not been initialized on " + device.toString());
        }
        return value;
    }

    public NDArray[] getAllValues(Parameter parameter) {
        NDArray[] allValues = new NDArray[devices.length];
        for (int i = 0; i < devices.length; i++) {
            allValues[i] = parameterValues.get(new Pair<>(devices[i].getDeviceId(), parameter));
        }
        return allValues;
    }

    public NDArray[] getAllGradients(Parameter parameter) {
        NDArray[] allGradients = new NDArray[devices.length];
        for (int i = 0; i < devices.length; i++) {
            allGradients[i] =
                    parameterValues
                            .get(new Pair<>(devices[i].getDeviceId(), parameter))
                            .getGradient();
        }
        return allGradients;
    }

    @Override
    public void close() {
        parameterValues.clear();
        parameterServer.close();
    }
}
