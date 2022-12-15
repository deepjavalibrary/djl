/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.util;

import java.util.concurrent.atomic.AtomicReference;

/**
 * {@code NativeResourceImpl} is an internal class for {@link AutoCloseable} blocks of memory
 * created in the different engines.
 *
 * @param <T> the resource that could map to a native pointer or java object
 */
public abstract class NativeResourceImpl<T> implements NativeResource<T> {

    protected final AtomicReference<T> handle;
    private String uid;

    protected NativeResourceImpl(T handle) {
        this.handle = new AtomicReference<>(handle);
        uid = handle.toString();
    }

    /** {@inheritDoc} */
    @Override
    public T getAndSetHandleNull() {
        return handle.getAndSet(null);
    }

    /** {@inheritDoc} */
    @Override
    public boolean isReleased() {
        return handle.get() == null;
    }

    /** {@inheritDoc} */
    @Override
    public T getHandle() {
        T reference = handle.get();
        if (reference == null) {
            throw new IllegalStateException("Native resource has been release already.");
        }
        return reference;
    }

    /** {@inheritDoc} */
    @Override
    public final String getUid() {
        return uid;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        throw new UnsupportedOperationException("Not implemented.");
    }
}
