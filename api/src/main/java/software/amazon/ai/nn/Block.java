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
package software.amazon.ai.nn;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.List;
import software.amazon.ai.Device;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.LayoutType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.util.PairList;

/** An interface defining neural-network layers. */
public interface Block {

    default Block asInDevice(Device device, boolean copy) {
        //  TODO: add implementation
        return this;
    }

    NDList forward(NDList inputs, PairList<String, Object> params);

    NDList forward(NDList inputs);

    void backward();

    void setInitializer(Initializer initializer);

    void setInitializer(Initializer initializer, String paramName);

    Shape[] initialize(NDManager manager, DataType dataType, Device[] devices, Shape[] inputShapes);

    boolean isInitialized();

    void cast(DataType dataType);

    void clear();

    DataDesc[] describeInput();

    PairList<String, Block> getChildren();

    List<Parameter> getDirectParameters();

    PairList<String, Parameter> getParameters();

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
