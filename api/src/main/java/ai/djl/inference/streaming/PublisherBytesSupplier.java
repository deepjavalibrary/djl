/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.inference.streaming;

import ai.djl.ndarray.BytesSupplier;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/**
 * An {@link PublisherBytesSupplier} is a streaming {@link BytesSupplier} suitable for reactive
 * asynchronous usage.
 */
public class PublisherBytesSupplier implements BytesSupplier {

    private final List<byte[]> allData;
    private final AtomicBoolean completed;
    private Consumer<byte[]> subscriber;
    private final AtomicInteger dataPushed;

    /** Constructs a {@link PublisherBytesSupplier}. */
    public PublisherBytesSupplier() {
        allData = new ArrayList<>();
        completed = new AtomicBoolean();
        dataPushed = new AtomicInteger();
    }

    /**
     * Appends content to the {@code BytesSupplier}.
     *
     * @param data bytes to append
     * @param lastChunk true if this is the last chunk
     */
    public void appendContent(byte[] data, boolean lastChunk) {
        synchronized (allData) {
            allData.add(data);
        }
        if (lastChunk) {
            completed.set(true);
        }
        pushData();
    }

    /**
     * Adds the subscriber to the {@link BytesSupplier} to get notified about additional data.
     *
     * @param subscriber a consumer function that will receive bytes when new daata is added and
     *     null when completed
     */
    public void subscribe(Consumer<byte[]> subscriber) {
        if (this.subscriber != null) {
            throw new IllegalStateException(
                    "The PublisherBytesSupplier only allows a single Subscriber");
        }
        this.subscriber = subscriber;
        pushData();
    }

    private void pushData() {
        if (subscriber == null) {
            return;
        }

        int dataAvailable;
        synchronized (allData) {
            dataAvailable = allData.size();
        }

        int sent = dataPushed.getAndSet(dataAvailable);
        if (sent < dataAvailable) {
            synchronized (this) {
                for (; sent < dataAvailable; sent++) {
                    subscriber.accept(allData.get(sent));
                }
                if (completed.get()) {
                    subscriber.accept(null);
                }
            }
        }
    }

    /** Waits until completed before passing thread (BLOCKS THREAD!). */
    @SuppressWarnings("PMD.EmptyControlStatement")
    public void waitToRead() {
        // Block until complete!!!
        while (!completed.get()) {
            // Do nothing
        }
    }

    /** {@inheritDoc} */
    @Override
    public byte[] getAsBytes() {
        if (!completed.get()) {
            throw new IllegalStateException(
                    "PublisherByteSupplier must be completely filled before reading.");
        }

        try (ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            for (byte[] data : allData) {
                bos.write(data);
            }
            return bos.toByteArray();
        } catch (IOException e) {
            throw new AssertionError("Failed to read BytesSupplier", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        return ByteBuffer.wrap(getAsBytes());
    }
}
