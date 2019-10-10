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
package ai.djl.test.mock;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.PairList;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.file.Path;

public class MockNDManager implements NDManager {

    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity);
    }

    @Override
    public NDArray create(Shape shape, DataType dataType, Device device) {
        return new MockNDArray(this, device, shape, dataType, null);
    }

    @Override
    public NDArray createCSR(
            Buffer data, long[] indptr, long[] indices, Shape shape, Device device) {
        return null;
    }

    @Override
    public NDArray createRowSparse(
            Buffer data, Shape dataShape, long[] indices, Shape shape, Device device) {
        return null;
    }

    @Override
    public NDList load(Path path) {
        return null;
    }

    @Override
    public void save(Path path, NDList ndList) {}

    @Override
    public NDArray zeros(Shape shape, DataType dataType, Device device) {
        return null;
    }

    @Override
    public NDArray ones(Shape shape, DataType dataType, Device device) {
        return null;
    }

    @Override
    public NDArray arange(int start, int stop, int step, DataType dataType, Device device) {
        return null;
    }

    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType, Device device) {
        return null;
    }

    @Override
    public NDArray linspace(double start, double stop, int num, boolean endPoint, Device device) {
        return null;
    }

    @Override
    public NDArray randomUniform(
            double low, double high, Shape shape, DataType dataType, Device device) {
        return null;
    }

    @Override
    public NDArray randomNormal(
            double loc, double scale, Shape shape, DataType dataType, Device device) {
        return null;
    }

    @Override
    public NDArray randomMultinomial(int n, NDArray pValues, Shape shape) {
        return null;
    }

    @Override
    public NDArray randomMultinomial(int n, NDArray pValues) {
        return null;
    }

    @Override
    public NDManager getParentManager() {
        return this;
    }

    @Override
    public Device getDevice() {
        return Device.defaultDevice();
    }

    @Override
    public NDManager newSubManager() {
        return this;
    }

    @Override
    public NDManager newSubManager(Device device) {
        return this;
    }

    @Override
    public void attach(String resourceId, AutoCloseable resource) {}

    @Override
    public void detach(String resourceId) {}

    @Override
    public void invoke(String operation, NDList src, NDList dest, PairList<String, ?> params) {}

    @Override
    public NDList invoke(String operation, NDList src, PairList<String, ?> params) {
        return null;
    }

    @Override
    public void close() {}
}
