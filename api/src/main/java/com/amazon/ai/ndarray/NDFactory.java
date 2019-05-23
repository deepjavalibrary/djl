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
package com.amazon.ai.ndarray;

import com.amazon.ai.Context;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.ndarray.types.SparseFormat;

public interface NDFactory extends AutoCloseable {

    NDArray create(
            Context context,
            Shape shape,
            DataType dataType,
            SparseFormat sparseFormat,
            boolean delay);

    NDArray create(DataDesc dataDesc);

    NDFactory getParentFactory();

    Context getContext();

    NDFactory newSubFactory();

    NDFactory newSubFactory(Context context);

    void attach(AutoCloseable resource);

    void detach(AutoCloseable resource);

    void close();
}
