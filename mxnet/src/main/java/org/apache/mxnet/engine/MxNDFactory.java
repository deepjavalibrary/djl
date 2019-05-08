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
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.ndarray.types.SparseFormat;
import com.sun.jna.Pointer;

public class MxNDFactory extends MxResourceAllocator implements NDFactory {

    private Context context;

    public MxNDFactory() {
        this(Context.defaultContext());
    }

    public MxNDFactory(Context context) {
        this.context = context;
    }

    @Override
    public NDArray create(
            Context context,
            Shape shape,
            DataType dataType,
            SparseFormat sparseFormat,
            boolean delay) {
        return new MxNDArray(this, context, shape, dataType, delay);
    }

    @Override
    public NDArray create(DataDesc dataDesc) {
        return create(
                dataDesc.getOrDefault(),
                dataDesc.getShape(),
                dataDesc.getDataType(),
                SparseFormat.DEFAULT,
                false);
    }

    public MxNDArray create(Pointer handle) {
        return new MxNDArray(this, context, SparseFormat.DEFAULT, null, DataType.FLOAT32, handle);
    }
}
