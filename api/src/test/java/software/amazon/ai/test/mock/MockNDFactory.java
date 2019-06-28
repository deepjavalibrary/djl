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
package software.amazon.ai.test.mock;

import java.nio.Buffer;
import software.amazon.ai.Context;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDFactory;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.PairList;

public class MockNDFactory implements NDFactory {

    @Override
    public NDArray create(Context context, Shape shape, DataType dataType) {
        return new MockNDArray();
    }

    @Override
    public NDArray create(DataDesc dataDesc) {
        return new MockNDArray();
    }

    @Override
    public NDArray create(DataDesc dataDesc, Buffer data) {
        return null;
    }

    @Override
    public void invoke(
            String operation, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {}

    @Override
    public NDArray[] invoke(String operation, NDArray[] src, PairList<String, ?> params) {
        return new NDArray[0];
    }

    @Override
    public NDArray zeros(Shape shape) {
        return null;
    }

    @Override
    public NDArray create(float[] data, Context context, Shape shape) {
        return null;
    }

    @Override
    public NDArray create(int[] data, Context context, Shape shape) {
        return null;
    }

    @Override
    public NDArray create(double[] data, Context context, Shape shape) {
        return null;
    }

    @Override
    public NDArray create(long[] data, Context context, Shape shape) {
        return null;
    }

    @Override
    public NDArray create(byte[] data, Context context, Shape shape) {
        return null;
    }

    @Override
    public NDArray zeros(Context context, Shape shape, DataType dataType) {
        return null;
    }

    @Override
    public NDArray zeros(DataDesc dataDesc) {
        return null;
    }

    @Override
    public NDArray ones(Context context, Shape shape, DataType dataType) {
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
