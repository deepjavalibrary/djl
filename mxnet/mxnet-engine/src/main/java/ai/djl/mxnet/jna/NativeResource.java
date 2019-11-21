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
package ai.djl.mxnet.jna;

import com.sun.jna.Pointer;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@code NativeResource} is an internal class for {@link AutoCloseable} blocks of memory created in
 * the MXNet Engine.
 */
public abstract class NativeResource implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(NativeResource.class);

    protected final AtomicReference<Pointer> handle;
    private String uid;
    private Exception exception;

    protected NativeResource(Pointer pointer) {
        this.handle = new AtomicReference<>(pointer);
        uid = String.valueOf(Pointer.nativeValue(pointer));
        if (logger.isTraceEnabled()) {
            exception = new Exception();
        }
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
    public Pointer getHandle() {
        Pointer pointer = handle.get();
        if (pointer == null) {
            throw new IllegalStateException("Native resource has been release already.");
        }
        return pointer;
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

    /** {@inheritDoc} */
    @SuppressWarnings("deprecation")
    @Override
    protected void finalize() throws Throwable {
        if (handle.get() != null) {
            if (exception != null) {
                logger.warn(
                        "Resource ({}) was not closed explicitly: {}",
                        getUid(),
                        getClass().getSimpleName());
                logger.warn("Resource was created:", exception);
            }
        }
        close();
        super.finalize();
    }
}
