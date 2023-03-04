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
package ai.djl.ndarray;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.PairList;
import ai.djl.util.RandomUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** {@code BaseNDManager} is the default implementation of {@link NDManager}. */
public abstract class BaseNDManager implements NDManager {

    private static final Logger logger = LoggerFactory.getLogger(BaseNDManager.class);

    protected NDManager parent;
    protected NDManager alternativeManager;
    protected String uid;
    protected String name;
    protected Device device;
    protected ConcurrentHashMap<String, AutoCloseable> resources;
    protected ConcurrentHashMap<String, TempResource> tempResources;
    protected AtomicBoolean closed = new AtomicBoolean(false);
    protected AtomicBoolean capped = new AtomicBoolean(false);

    protected BaseNDManager(NDManager parent, Device device) {
        this.parent = parent;
        this.device = device == null ? defaultDevice() : device;
        resources = new ConcurrentHashMap<>();
        tempResources = new ConcurrentHashMap<>();
        uid = UUID.randomUUID().toString();
        Engine engine = getEngine().getAlternativeEngine();
        if (engine != null) {
            alternativeManager = engine.newBaseManager(Device.cpu());
        }
    }

    /** {@inheritDoc} */
    @Override
    public final Device defaultDevice() {
        return getEngine().defaultDevice();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(String[] data, Charset charset, Shape shape) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(Shape shape, DataType dataType) {
        return zeros(shape, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createCSR(Buffer data, long[] indptr, long[] indices, Shape shape) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createRowSparse(Buffer data, Shape dataShape, long[] indices, Shape shape) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createCoo(Buffer data, long[][] indices, Shape shape) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDList load(Path path) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public void setName(String name) {
        this.name = name;
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return this.name == null ? uid : this.name;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray full(Shape shape, float value, DataType dataType) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray arange(float start, float stop, float step, DataType dataType) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray linspace(float start, float stop, int num, boolean endpoint) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomInteger(long low, long high, Shape shape, DataType dataType) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomPermutation(long n) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomUniform(float low, float high, Shape shape, DataType dataType) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(float loc, float scale, Shape shape, DataType dataType) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray truncatedNormal(float loc, float scale, Shape shape, DataType dataType) {
        int sampleSize = (int) shape.size();
        double[] dist = new double[sampleSize];

        for (int i = 0; i < sampleSize; i++) {
            double sample = RandomUtils.nextGaussian();
            while (sample < -2 || sample > 2) {
                sample = RandomUtils.nextGaussian();
            }

            dist[i] = sample;
        }

        return create(dist).muli(scale).addi(loc).reshape(shape).toType(dataType, false);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues, Shape shape) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sampleNormal(NDArray mu, NDArray sigma) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sampleNormal(NDArray mu, NDArray sigma, Shape shape) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray samplePoisson(NDArray lam) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray samplePoisson(NDArray lam, Shape shape) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sampleGamma(NDArray alpha, NDArray beta) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sampleGamma(NDArray alpha, NDArray beta, Shape shape) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public boolean isOpen() {
        return !closed.get();
    }

    /** {@inheritDoc} */
    @Override
    public void cap() {
        this.capped.set(true);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getParentManager() {
        return parent;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newSubManager() {
        return newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        return device;
    }

    /** {@inheritDoc} */
    @Override
    public List<NDArray> getManagedArrays() {
        return Stream.concat(
                        // Main resources
                        resources.values().stream()
                                .flatMap(
                                        r -> {
                                            if (r instanceof NDResource) {
                                                return ((NDResource) r)
                                                        .getResourceNDArrays().stream();
                                            } else if (r instanceof NDManager) {
                                                return ((NDManager) r).getManagedArrays().stream();
                                            } else {
                                                return Stream.empty();
                                            }
                                        }),

                        // Temp resouces
                        tempResources.values().stream()
                                .flatMap(tr -> tr.resource.getResourceNDArrays().stream()))
                .collect(Collectors.toList());
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        String parentName = parent == null ? "No Parent" : parent.getName();
        return "Name: "
                + getName()
                + " Parent Name: "
                + parentName
                + " isOpen: "
                + isOpen()
                + " Resource size: "
                + resources.size();
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void attachInternal(String resourceId, AutoCloseable resource) {
        if (capped.get()) {
            throw new IllegalStateException("NDManager is capped for addition of resources.");
        }
        attachUncappedInternal(resourceId, resource);
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void attachUncappedInternal(String resourceId, AutoCloseable resource) {
        if (closed.get()) {
            throw new IllegalStateException("NDManager has been closed already.");
        }
        tempResources.compute(
                resourceId,
                (key, tempResource) -> {
                    if (tempResource != null) {
                        // This state occurs when this manager (manA) tempAttaches a resource that
                        // is later
                        // tempAttached to another manager (manB)
                        // When manB is closed, it will use attach to return the resource to this
                        // (manA)
                        // In that case, it should stay as a tempResource in this (manA)
                        tempResource.detached = false;
                    } else {
                        resources.put(resourceId, resource);
                    }
                    return tempResource;
                });
    }

    /** {@inheritDoc} */
    @Override
    public void tempAttachInternal(
            NDManager originalManager, String resourceId, NDResource resource) {
        if (this instanceof SystemNDManager) {
            throw new IllegalStateException(
                    "System manager cannot be temp attached because it can't be closed..");
        }
        if (closed.get()) {
            throw new IllegalStateException("NDManager has been closed already.");
        }
        tempResources.put(resourceId, new TempResource(resource, originalManager));
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void detachInternal(String resourceId) {
        if (closed.get()) {
            // This may happen in the middle of BaseNDManager.close()
            return;
        }
        tempResources.computeIfPresent(
                resourceId,
                (key, tempResource) -> {
                    tempResource.detached = true;
                    return tempResource;
                });
        resources.remove(resourceId);
    }

    /** {@inheritDoc} */
    @Override
    public void invoke(
            String operation, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDList invoke(String operation, NDList src, PairList<String, ?> params) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (this instanceof SystemNDManager) {
            throw new IllegalStateException(
                    "The SystemNDManager can not be closed. It is global and lives for the duration"
                            + " of the process");
        }
        if (!closed.getAndSet(true)) {
            for (AutoCloseable closeable : resources.values()) {
                try {
                    closeable.close();
                } catch (Exception e) {
                    logger.error("Resource close failed.", e);
                }
            }
            for (TempResource resource : tempResources.values()) {
                resource.returnResource();
            }
            parent.detachInternal(uid);
            resources.clear();
            tempResources.clear();
        }
    }

    /**
     * Prints information about this {@link NDManager} and all sub-managers to the console.
     *
     * @param level the level of this {@link NDManager} in the hierarchy
     */
    public void debugDump(int level) {
        StringBuilder sb = new StringBuilder(100);
        for (int i = 0; i < level; ++i) {
            sb.append("    ");
        }
        sb.append("\\--- NDManager(")
                .append(uid.substring(24))
                .append(") resource count: ")
                .append(resources.size());

        System.out.println(sb); // NOPMD
        for (AutoCloseable c : resources.values()) {
            if (c instanceof BaseNDManager) {
                ((BaseNDManager) c).debugDump(level + 1);
            }
        }
    }

    NDManager getAlternativeManager() {
        return alternativeManager;
    }

    /**
     * Checks if the input buffer size is match expected data type.
     *
     * @param buffer the input buffer
     * @param dataType the desired {@code DataType}
     * @param expected the expected size
     * @throws IllegalArgumentException if buffer size is invalid
     */
    public static void validateBuffer(Buffer buffer, DataType dataType, int expected) {
        boolean isByteBuffer = buffer instanceof ByteBuffer;
        DataType type = DataType.fromBuffer(buffer);
        if (type != dataType && !isByteBuffer) {
            // It's ok if type != datatype and buffer is ByteBuffer,
            // since buffer will be copied into ByteBuffer
            throw new IllegalArgumentException(
                    "The input data type: "
                            + type
                            + " does not match target array data type: "
                            + dataType);
        }

        int remaining = buffer.remaining();
        int expectedSize = isByteBuffer ? dataType.getNumOfBytes() * expected : expected;
        if (remaining < expectedSize) {
            throw new IllegalArgumentException(
                    "The NDArray size is: " + expected + ", but buffer size is: " + remaining);
        }
        if (remaining > expectedSize) {
            logger.warn(
                    "Input buffer size is greater than the NDArray size, please set limit"
                            + " explicitly.");
            buffer.limit(expectedSize);
        }
    }

    /**
     * Copies data from the source {@code Buffer} to the target {@code ByteBuffer}.
     *
     * @param src the source {@code Buffer}
     * @param target the target {@code ByteBuffer}
     */
    public static void copyBuffer(Buffer src, ByteBuffer target) {
        target.rewind();
        DataType inputType = DataType.fromBuffer(src);
        switch (inputType) {
            case FLOAT16:
                target.asShortBuffer().put((ShortBuffer) src);
                break;
            case FLOAT32:
                target.asFloatBuffer().put((FloatBuffer) src);
                break;
            case FLOAT64:
                target.asDoubleBuffer().put((DoubleBuffer) src);
                break;
            case UINT8:
            case INT8:
            case BOOLEAN:
                target.put((ByteBuffer) src);
                break;
            case INT32:
                target.asIntBuffer().put((IntBuffer) src);
                break;
            case INT64:
                target.asLongBuffer().put((LongBuffer) src);
                break;
            default:
                throw new AssertionError("Unsupported datatype: " + inputType);
        }
        target.rewind();
    }

    protected static final class TempResource {

        private NDResource resource;
        private NDManager manager;
        private boolean detached;

        public TempResource(NDResource resource, NDManager manager) {
            this.resource = resource;
            this.manager = manager;
            this.detached = false;
        }

        public void returnResource() {
            try {
                if (!detached) {
                    if (manager.isOpen()) {
                        resource.returnResource(manager);
                    } else {
                        resource.close();
                    }
                }
            } catch (Exception e) {
                logger.error("Temporary resource return failed.", e);
            }
        }
    }
}
