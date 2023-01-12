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
package ai.djl.ndarray.refcount;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.ref.PhantomReference;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A subclass of {@link PhantomReference} that also acts as a linked list to keep their references
 * alive until they get garbage collected. Implements reference counting with an {@link
 * AtomicInteger} count.
 *
 * <p>This class has been derived from {@code org.bytedeco.javacpp.Pointer} by Samuel Audet
 */
@SuppressWarnings({"PMD.MutableStaticState", "PMD.AvoidUsingVolatile"})
public class DeallocatorReference extends PhantomReference<RCObject>
        implements Deallocator, ReferenceCounter {

    private static final Logger logger = LoggerFactory.getLogger(DeallocatorReference.class);
    static volatile DeallocatorReference head;
    static volatile long totalCount;
    volatile DeallocatorReference prev = this;
    volatile DeallocatorReference next = this;
    protected Deallocator deallocator;
    AtomicInteger count;

    protected DeallocatorReference(RCObject p, Deallocator deallocator) {
        super(p, null);
        this.deallocator = deallocator;
        this.count = new AtomicInteger(0);
    }

    final void add() {
        synchronized (DeallocatorReference.class) {
            if (head == null) {
                head = this;
                prev = next = null;
            } else {
                prev = null;
                next = head;
                next.prev = head = this;
            }
            totalCount++;
        }
    }

    final void remove() {
        if (prev == this && next == this) {
            return;
        }
        synchronized (DeallocatorReference.class) {
            if (prev == null) {
                head = next;
            } else {
                prev.next = next;
            }
            if (next != null) {
                next.prev = prev;
            }
            prev = next = this;
            totalCount--;
        }
    }

    /** {@inheritDoc} */
    @Override
    public void clear() {
        super.clear();
        if (deallocator != null) {
            if (logger.isDebugEnabled()) {
                logger.trace("Collecting " + this);
            }
            deallocate();
        }
    }

    /** {@inheritDoc} */
    @Override
    public void deallocate() {
        if (deallocator != null) {
            deallocator.deallocate();
            deallocator = null;
        }
    }

    /** {@inheritDoc} */
    @Override
    public void retain() {
        if (deallocator != null) {
            count.incrementAndGet();
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean release() {
        if (deallocator != null && count.decrementAndGet() <= 0) {
            if (logger.isDebugEnabled()) {
                logger.trace("Releasing " + this);
            }
            deallocate();
            return true;
        }
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public int count() {
        return deallocator != null ? count.get() : -1;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return getClass().getName() + "[deallocator=" + deallocator + ",count=" + count + "]";
    }
}
