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

import java.util.List;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.util.PairList;

/** An interface defining neural-network layers. */
public interface Block {

    Block IDENTITY_BLOCK = new LambdaBlock(x -> x);

    NDList forward(NDList inputs, PairList<String, Object> params);

    NDList forward(NDList inputs);

    void backward();

    Shape getOutputShape(Shape... inputs);

    List<Parameter> getDirectParameters();

    void setInitializer(NDManager manager, Initializer initializer);

    void setInitializer(NDManager manager, Initializer initializer, boolean overwrite);

    void setInitializer(NDManager manager, Initializer initializer, String paramName);

    void setInitializer(
            NDManager manager, Initializer initializer, String paramName, boolean overwrite);

    // TODO: remove this API
    void ensureInitialized(NDList inputs);

    Block cast(DataType dataType);

    DataDesc[] describeInput();

    Shape getParameterShape(String name, NDList inputs);

    byte[] getEncoded();

    PairList<String, Block> getChildren();

    PairList<String, Parameter> getParameters();

    PairList<String, Parameter> getChildrenParameters();
}
