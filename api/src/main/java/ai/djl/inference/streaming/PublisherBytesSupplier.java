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

import java.nio.ByteBuffer;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * An {@link PublisherBytesSupplier} is a streaming {@link BytesSupplier} suitable for reactive
 * asynchronous usage.
 */
public class PublisherBytesSupplier implements BytesSupplier {

    private Consumer<byte[]> subscriber;
    private CountDownLatch latch;
    private CompletableFuture<Void> future;

    /** Constructs a {@link PublisherBytesSupplier}. */
    public PublisherBytesSupplier() {
        latch = new CountDownLatch(1);
        future = new CompletableFuture<>();
    }

    /**
     * Appends content to the {@code BytesSupplier}.
     *
     * @param data bytes to append
     * @param lastChunk true if this is the last chunk
     */
    public void appendContent(byte[] data, boolean lastChunk) {
        if (subscriber == null) {
            try {
                if (!latch.await(2, TimeUnit.MINUTES)) {
                    throw new IllegalStateException("Wait for subscriber timeout.");
                }
                if (subscriber == null) {
                    // workaround Spotbugs
                    throw new IllegalStateException("subscriber is not set.");
                }
            } catch (InterruptedException e) {
                throw new IllegalStateException("Append content interrupted.", e);
            }
        }
        subscriber.accept(data);
        if (lastChunk) {
            subscriber.accept(null);
            future.complete(null);
        }
    }

    /**
     * Adds the subscriber to the {@link BytesSupplier} to get notified about additional data.
     *
     * @param subscriber a consumer function that will receive bytes when new daata is added and
     *     null when completed
     * @return a {@code CompletableFuture} object
     */
    public CompletableFuture<Void> subscribe(Consumer<byte[]> subscriber) {
        if (this.subscriber != null) {
            throw new IllegalStateException(
                    "The PublisherBytesSupplier only allows a single Subscriber");
        }
        this.subscriber = subscriber;
        latch.countDown();
        return future;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        throw new UnsupportedOperationException("Not supported.");
    }
}
