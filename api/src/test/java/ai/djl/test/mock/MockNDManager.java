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
import ai.djl.engine.Engine;
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

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(Shape shape, DataType dataType) {
        return new MockNDArray(this, getDevice(), shape, dataType, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createCSR(Buffer data, long[] indptr, long[] indices, Shape shape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createRowSparse(Buffer data, Shape dataShape, long[] indices, Shape shape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList load(Path path) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Shape shape, DataType dataType) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray full(Shape shape, float value, DataType dataType) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray arange(float start, float stop, float step, DataType dataType) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray linspace(float start, float stop, int num, boolean endPoint) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomUniform(float low, float high, Shape shape, DataType dataType) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(float loc, float scale, Shape shape, DataType dataType) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues, Shape shape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isOpen() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getParentManager() {
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        return Device.cpu();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newSubManager() {
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newSubManager(Device device) {
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public void attach(String resourceId, AutoCloseable resource) {}

    /** {@inheritDoc} */
    @Override
    public void detach(String resourceId) {}

    /** {@inheritDoc} */
    @Override
    public void invoke(
            String operation, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {}

    /** {@inheritDoc} */
    @Override
    public NDList invoke(String operation, NDList src, PairList<String, ?> params) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {}
}
