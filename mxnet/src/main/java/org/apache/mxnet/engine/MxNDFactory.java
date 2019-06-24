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
import com.amazon.ai.util.PairList;
import com.sun.jna.Pointer;
import java.nio.Buffer;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.mxnet.jna.JnaUtils;

public class MxNDFactory implements NDFactory {

    /**
     * A global {@link NDFactory} singleton instance.
     *
     * <p>This NDFactory is the root of all the other NDFactories. NDArrays created by this factory
     * are un-managed, user has to close them manually. Those NDArrays will be released on GC, and
     * might be run into out of native memory issue.
     */
    static final MxNDFactory SYSTEM_FACTORY = new SystemFactory();

    private static final NDArray[] EMPTY = new NDArray[0];

    private NDFactory parent;
    private Context context;
    private Map<AutoCloseable, AutoCloseable> resources;

    public static MxNDFactory getSystemFactory() {
        return SYSTEM_FACTORY;
    }

    private MxNDFactory(NDFactory parent, Context context) {
        this.parent = parent;
        this.context = context;
        resources = new ConcurrentHashMap<>();
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray create(Context context, Shape shape, DataType dataType) {
        if (context == null) {
            context = this.context;
        }
        MxNDArray array = new MxNDArray(this, context, shape, dataType);
        resources.put(array, array);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public Context getContext() {
        return context;
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray create(DataDesc dataDesc) {
        return create(dataDesc.getContext(), dataDesc.getShape(), dataDesc.getDataType());
    }

    @Override
    public MxNDArray create(float[] data, Context context, Shape shape) {
        MxNDArray mxNDArray = create(context, shape, DataType.FLOAT32);
        mxNDArray.set(data);
        return mxNDArray;
    }

    @Override
    public MxNDArray create(int[] data, Context context, Shape shape) {
        MxNDArray mxNDArray = create(context, shape, DataType.INT32);
        mxNDArray.set(data);
        return mxNDArray;
    }

    @Override
    public MxNDArray create(double[] data, Context context, Shape shape) {
        MxNDArray mxNDArray = create(context, shape, DataType.FLOAT64);
        mxNDArray.set(data);
        return mxNDArray;
    }

    @Override
    public MxNDArray create(long[] data, Context context, Shape shape) {
        MxNDArray mxNDArray = create(context, shape, DataType.INT64);
        mxNDArray.set(data);
        return mxNDArray;
    }

    @Override
    public MxNDArray create(byte[] data, Context context, Shape shape) {
        MxNDArray mxNDArray = create(context, shape, DataType.INT8);
        mxNDArray.set(data);
        return mxNDArray;
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray create(DataDesc dataDesc, Buffer data) {
        MxNDArray array =
                create(dataDesc.getContext(), dataDesc.getShape(), dataDesc.getDataType());
        array.set(data);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray[] invoke(
            String operation, NDArray[] src, NDArray[] dest, PairList<String, String> params) {
        return JnaUtils.op(operation).invoke(this, src, dest, params);
    }

    public NDArray invoke(String operation, NDArray src, PairList<String, String> params) {
        if (src == null) {
            return invoke(operation, EMPTY, null, params)[0];
        }
        return invoke(operation, new NDArray[] {src}, null, params)[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape) {
        return zeros(context, shape, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Context context, Shape shape, DataType dataType) {
        return fill("_zeros", context, shape, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(DataDesc dataDesc) {
        return zeros(dataDesc.getContext(), dataDesc.getShape(), dataDesc.getDataType());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Context context, Shape shape, DataType dataType) {
        return fill("_ones", context, shape, dataType);
    }

    private NDArray fill(String opName, Context context, Shape shape, DataType dataType) {
        MxOpParams params = new MxOpParams();
        if (shape == null) {
            throw new IllegalArgumentException("Shape is required for " + opName.substring(1));
        }
        params.setShape(shape);
        params.setContext(context);
        params.setDataType(dataType);
        return invoke(opName, EMPTY, null, params)[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(DataDesc dataDesc) {
        return ones(dataDesc.getContext(), dataDesc.getShape(), dataDesc.getDataType());
    }

    public MxNDArray create(Pointer handle) {
        MxNDArray array = new MxNDArray(this, null, null, null, handle);
        resources.put(array, array);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDFactory getParentFactory() {
        return parent;
    }

    /** {@inheritDoc} */
    @Override
    public MxNDFactory newSubFactory() {
        return newSubFactory(context);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDFactory newSubFactory(Context context) {
        MxNDFactory factory = new MxNDFactory(this, context);
        resources.put(factory, factory);
        return factory;
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void attach(AutoCloseable resource) {
        resources.put(resource, resource);
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void detach(AutoCloseable resource) {
        resources.remove(resource);
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void close() {
        for (AutoCloseable resource : resources.keySet()) {
            try {
                resource.close();
            } catch (Exception ignore) {
                // ignore
            }
        }
        resources = null;
        parent.detach(this);
    }

    private static final class SystemFactory extends MxNDFactory {

        SystemFactory() {
            super(null, Context.defaultContext());
        }

        /** {@inheritDoc} */
        @Override
        public void attach(AutoCloseable resource) {}

        /** {@inheritDoc} */
        @Override
        public void detach(AutoCloseable resource) {}

        /** {@inheritDoc} */
        @Override
        public void close() {}
    }
}
