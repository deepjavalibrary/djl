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
package org.apache.mxnet.model;

import com.sun.jna.Pointer;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class NativeResource implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(NativeResource.class);

    protected final AtomicReference<Pointer> handle;
    protected final ResourceAllocator alloc;
    private Exception exception;

    protected NativeResource(ResourceAllocator alloc, Pointer pointer) {
        this.alloc = alloc;
        this.handle = new AtomicReference<>(pointer);
        if (alloc != null) {
            alloc.attach(this);
        }
        if (logger.isDebugEnabled()) {
            exception = new Exception();
        }
    }

    public Pointer getHandle() {
        Pointer pointer = handle.get();
        if (pointer == null) {
            throw new IllegalStateException("Native resource has been release already.");
        }
        return pointer;
    }

    @Override
    public void close() {
        throw new UnsupportedOperationException("Not implemented.");
    }

    @Override
    protected void finalize() throws Throwable {
        if (getHandle() != null) {
            logger.warn("Resource was not closed explicitly: {}", getClass().getSimpleName());
            logger.warn("Resource was created:", exception);
        }
        close();
        super.finalize();
    }
}
