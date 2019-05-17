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
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.ndarray.types.SparseFormat;
import com.sun.jna.Pointer;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class MxNDFactory implements NDFactory {

    public static final MxNDFactory SYSTEM_FACTORY = new SystemFactory();

    private NDFactory parent;
    private Context context;
    private Map<AutoCloseable, AutoCloseable> resources;

    private MxNDFactory(NDFactory parent, Context context) {
        this.parent = parent;
        this.context = context;
        resources = new ConcurrentHashMap<>();
    }

    @Override
    public MxNDArray create(
            Context context,
            Shape shape,
            DataType dataType,
            SparseFormat sparseFormat,
            boolean delay) {
        MxNDArray array = new MxNDArray(this, context, shape, dataType, delay);
        resources.put(array, array);
        return array;
    }

    @Override
    public MxNDArray create(DataDesc dataDesc) {
        return create(
                dataDesc.getOrDefault(),
                dataDesc.getShape(),
                dataDesc.getDataType(),
                SparseFormat.DEFAULT,
                false);
    }

    public MxNDArray create(Pointer handle) {
        MxNDArray array =
                new MxNDArray(this, context, SparseFormat.DEFAULT, null, DataType.FLOAT32, handle);
        resources.put(array, array);
        return array;
    }

    @Override
    public NDFactory getParentFactory() {
        return parent;
    }

    @Override
    public MxNDFactory newSubFactory() {
        return newSubFactory(context);
    }

    @Override
    public MxNDFactory newSubFactory(Context context) {
        MxNDFactory factory = new MxNDFactory(this, context);
        resources.put(factory, factory);
        return factory;
    }

    @Override
    public synchronized void attach(AutoCloseable resource) {
        resources.put(resource, resource);
    }

    @Override
    public synchronized void detach(AutoCloseable resource) {
        resources.remove(resource);
    }

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

        @Override
        public void attach(AutoCloseable resource) {}

        @Override
        public void detach(AutoCloseable resource) {}

        @Override
        public void close() {}
    }
}
