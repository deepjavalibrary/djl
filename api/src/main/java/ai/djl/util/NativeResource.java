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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDScope;

import com.sun.jna.Pointer;

import java.text.MessageFormat;
import java.time.Instant;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReference;

/**
 * {@code NativeResource} is an internal class for {@link AutoCloseable} blocks of memory created in
 * the different engines.
 *
 * @param <T> the resource that could map to a native pointer or java object
 */
public abstract class NativeResource<T> implements AutoCloseable {

    protected final AtomicReference<T> handle;
    protected Instant creationTime;

    protected String uid;
    private String creationStackTraceAsString;
    private String closingStackTraceAsString;

    /** Constructs a new {@code NativeResource}. */
    public NativeResource() {
        //  super();
        handle = new AtomicReference<>();
        this.creationTime = Instant.now();
        if (NDScope.isVerboseIfResourceAlreadyClosed()) {
            creationStackTraceAsString = stackTraceAsString();
        }
    }

    protected NativeResource(T handle) {
        this.handle = new AtomicReference<>(handle);
        uid = handle.toString();
        this.creationTime = Instant.now();
        if (NDScope.isVerboseIfResourceAlreadyClosed()) {
            creationStackTraceAsString = stackTraceAsString();
        }
    }

    private String fingerPrintOfNDArrayWithStackTraces() {
        String name = "NO_NAME";
        if (this instanceof NDArray) {
            name = ((NDArray) this).getName();
        }
        return MessageFormat.format(
                "NDArray named \"{0}\" identified by (uid:{1};createdAt:{2}) \n"
                        + "call stack at creation...{3}\n"
                        + "######### \n"
                        + "call stack at closing...{4}\n"
                        + "#########",
                name,
                getUid(),
                creationTime,
                creationStackTraceAsString,
                closingStackTraceAsString);
    }

    /**
     * Returns the current stack trace as a string.
     *
     * @return the current stack trace as a string
     */
    public static String stackTraceAsString() {
        StringBuilder buf = new StringBuilder();
        Arrays.stream(Thread.currentThread().getStackTrace())
                .forEach(
                        s ->
                                buf.append(
                                        "\nat "
                                                + s.getClassName()
                                                + "."
                                                + s.getMethodName()
                                                + "("
                                                + s.getFileName()
                                                + ":"
                                                + s.getLineNumber()
                                                + ")"));
        return buf.toString();
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
            String message = "Native resource has been released already. ";
            if (NDScope.isVerboseIfResourceAlreadyClosed() && this instanceof NDArray) {
                message += fingerPrintOfNDArrayWithStackTraces();
            }
            throw new IllegalStateException(message);
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

    /**
     * Remembers the stack trace on closing an NDArray if {@link
     * NDScope#isVerboseIfResourceAlreadyClosed()} is set.
     */
    public void onClose() {
        if (NDScope.isVerboseIfResourceAlreadyClosed()) {
            this.closingStackTraceAsString = stackTraceAsString();
        }
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        throw new UnsupportedOperationException("Not implemented.");
    }
}
