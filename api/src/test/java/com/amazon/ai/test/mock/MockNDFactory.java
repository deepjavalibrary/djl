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
package com.amazon.ai.test.mock;

import com.amazon.ai.Context;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.ndarray.types.SparseFormat;
import com.amazon.ai.util.PairList;

public class MockNDFactory implements NDFactory {

    @Override
    public NDArray create(
            Context context, Shape shape, DataType dataType, SparseFormat sparseFormat) {
        return new MockNDArray();
    }

    @Override
    public NDArray create(DataDesc dataDesc) {
        return new MockNDArray();
    }

    @Override
    public NDArray[] invoke(
            String operation, NDArray[] src, NDArray[] dest, PairList<String, String> params) {
        return new NDArray[0];
    }

    @Override
    public NDArray zeros(Shape shape) {
        return null;
    }

    @Override
    public NDArray zeros(
            Context context, Shape shape, DataType dataType, SparseFormat sparseFormat) {
        return null;
    }

    @Override
    public NDArray zeros(DataDesc dataDesc) {
        return null;
    }

    @Override
    public NDArray ones(
            Context context, Shape shape, DataType dataType, SparseFormat sparseFormat) {
        return null;
    }

    @Override
    public NDArray ones(DataDesc dataDesc) {
        return null;
    }

    @Override
    public NDFactory getParentFactory() {
        return this;
    }

    @Override
    public Context getContext() {
        return Context.defaultContext();
    }

    @Override
    public NDFactory newSubFactory() {
        return this;
    }

    @Override
    public NDFactory newSubFactory(Context context) {
        return this;
    }

    @Override
    public void attach(AutoCloseable resource) {}

    @Override
    public void detach(AutoCloseable resource) {}

    @Override
    public void close() {}
}
