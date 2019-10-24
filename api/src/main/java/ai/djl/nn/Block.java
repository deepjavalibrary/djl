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
package ai.djl.nn;

import ai.djl.Device;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.List;

/** An interface defining neural-network layers. */
public interface Block {

    default NDList forward(ParameterStore parameterStore, NDList inputs) {
        return forward(parameterStore, inputs, null);
    }

    NDList forward(ParameterStore parameterStore, NDList inputs, PairList<String, Object> params);

    void setInitializer(Initializer initializer);

    void setInitializer(Initializer initializer, String paramName);

    Shape[] initialize(NDManager manager, DataType dataType, Device[] devices, Shape[] inputShapes);

    boolean isInitialized();

    void cast(DataType dataType);

    void clear();

    PairList<String, Shape> describeInput();

    BlockList getChildren();

    List<Parameter> getDirectParameters();

    ParameterList getParameters();

    Shape getParameterShape(String name, Shape[] inputShapes);

    Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes);

    void saveParameters(DataOutputStream os) throws IOException;

    void loadParameters(NDManager manager, DataInputStream is) throws IOException;

    static void validateLayout(LayoutType[] expectedLayout, LayoutType[] actualLayout) {
        if (actualLayout.length != expectedLayout.length) {
            throw new UnsupportedOperationException(
                    "Expected layout: "
                            + LayoutType.toString(expectedLayout)
                            + ", but got: "
                            + LayoutType.toString(actualLayout));
        }
        for (int i = 0; i < actualLayout.length; i++) {
            if (actualLayout[i] != LayoutType.UNKNOWN && actualLayout[i] != expectedLayout[i]) {
                throw new UnsupportedOperationException(
                        "Expected layout: "
                                + LayoutType.toString(expectedLayout)
                                + ", but got: "
                                + LayoutType.toString(actualLayout));
            }
        }
    }
}
