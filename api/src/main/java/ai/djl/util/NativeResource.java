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

import com.sun.jna.Pointer;
import java.util.concurrent.atomic.AtomicReference;

/**
 * {@code NativeResource} is an internal class for {@link AutoCloseable} blocks of memory created in
 * the different engines.
 *
 * @param <T> the resource that could map to a native pointer or java object
 */
public abstract class NativeResource<T> implements AutoCloseable {

    protected final AtomicReference<T> handle;
    private String uid;

    protected NativeResource(T handle) {
        this.handle = new AtomicReference<>(handle);
        uid = handle.toString();
    }

    /**
     * Gets the boolean that indicates whether this resource has been released.
     *
     * @return whether this resource has been released
     */
    public boolean isReleased() {
        return handle.get() == null;
    }

    /**
     * Gets the {@link Pointer} to this resource.
     *
     * @return the {@link Pointer} to this resource
     */
    public T getHandle() {
        T reference = handle.get();
        if (reference == null) {
            throw new IllegalStateException("Native resource has been release already.");
        }
        return reference;
    }

    /**
     * Gets the unique ID of this resource.
     *
     * @return the unique ID of this resource
     */
    public final String getUid() {
        return uid;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        throw new UnsupportedOperationException("Not implemented.");
    }
}
