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

import java.util.Iterator;

/**
 * All peer classes to native types must be descended from {@link RCObject} (reference counted
 * object), the topmost class.
 *
 * <p>It is also possible to use a {@link RCScope} to keep track of a group of {@link RCObject}
 * objects, and have them deallocated in a transparent but deterministic manner.
 *
 * <p>This class has been derived from {@code org.bytedeco.javacpp.Pointer} by Samuel Audet
 */
@SuppressWarnings("PMD.AvoidBranchingStatementAsLastInLoop")
public class RCObject {

    private static final Logger logger = LoggerFactory.getLogger(RCObject.class);

    private Deallocator deallocator;

    /**
     * Returns {@link DeallocatorReference#totalCount}, current number of ReferenceCountedObjects
     * tracked by deallocators.
     *
     * @return the total count
     */
    public static long totalCount() {
        return DeallocatorReference.totalCount;
    }

    /**
     * Returns {@link ReferenceCounter#count()} or -1 if no deallocator has been set.
     *
     * @return the count
     */
    public int referenceCount() {
        ReferenceCounter r = (ReferenceCounter) deallocator;
        return r != null ? r.count() : -1;
    }

    /**
     * Returns {@link #deallocator}.
     *
     * @return the deallocator
     */
    protected Deallocator deallocator() {
        return deallocator;
    }

    /**
     * Sets the deallocator and returns this. Also clears current deallocator if not {@code null}.
     * That is, it deallocates previously allocated memory. Should not be called more than once
     * after allocation.
     *
     * @param deallocator the new deallocator
     * @param <P> the type of the referenceCountedObject
     * @return this referenceCountedObject
     */
    protected <P extends RCObject> P deallocator(Deallocator deallocator) {
        if (this.deallocator != null) {
            if (logger.isDebugEnabled()) {
                logger.debug("Predeallocating " + this);
            }
            this.deallocator.deallocate();
            this.deallocator = null;
        }
        if (deallocator != null) {
            DeallocatorReference r =
                    deallocator instanceof DeallocatorReference
                            ? (DeallocatorReference) deallocator
                            : new DeallocatorReference(this, deallocator);
            this.deallocator = r;
            Iterator<RCScope> it = RCScope.getScopeIterator();
            if (it != null) {
                while (it.hasNext()) {
                    try {
                        it.next().attach(this);
                    } catch (IllegalArgumentException e) {
                        // try the next scope down the stack
                        continue;
                    }
                    break;
                }
            }
        }

        @SuppressWarnings("unchecked")
        P p = (P) this;
        return p;
    }

    /** Calls {@code deallocate(true)}. */
    public void deallocate() {
        deallocate(true);
    }

    /**
     * Explicitly manages native memory without waiting after the garbage collector. Has no effect
     * if no deallocator was previously set with {@link #deallocator(Deallocator)}.
     *
     * @param deallocate if true, deallocates, else does not, but disables garbage collection
     */
    public void deallocate(boolean deallocate) {
        DeallocatorReference r = (DeallocatorReference) deallocator;
        if (deallocate && deallocator != null) {
            if (logger.isDebugEnabled()) {
                logger.debug("Deallocating " + this);
            }
            deallocator.deallocate();
            deallocator = null;
            //  address = 0;
        }
        if (r != null) {
            // remove from queue without calling the deallocator
            r.deallocator = null;
            r.clear();
            r.remove();
            r.deallocator = deallocator;
        }
    }

    /**
     * Calls {@link ReferenceCounter#retain()}, incrementing the reference count by 1. Has no effect
     * if no deallocator was previously set with {@link #deallocator(Deallocator)}.
     *
     * @param <P> the type of the referenceCountedObject
     * @return this
     */
    public <P extends RCObject> P retainReference() {
        ReferenceCounter r = (ReferenceCounter) deallocator;
        if (r != null) {
            r.retain();
        }
        @SuppressWarnings("unchecked")
        P p = (P) this;
        return p;
    }

    /**
     * Calls {@link ReferenceCounter#release()}, decrementing the reference count by 1, in turn
     * deallocating this referenceCountedObject when the count drops to 0. Has no effect if no
     * deallocator was previously set with {@link #deallocator(Deallocator)}.
     *
     * @return true when the count drops to 0 and deallocation has occurred
     */
    public boolean releaseReference() {
        DeallocatorReference r = (DeallocatorReference) deallocator;
        if (r != null && r.release()) {
            deallocator = null;
            //  address = 0;
            r.clear();
            r.remove();
            return true;
        }
        return false;
    }

    /**
     * Calls in effect {@code memcpy(this.address + this.position, p.address + p.position, length)},
     * where {@code length = sizeof(p) * (p.limit - p.position)}. If limit == 0, it uses position +
     * 1 instead. The way the methods were designed allows constructs such as {@code
     * this.position(0).put(p.position(13).limit(42))}.
     *
     * @param p the referenceCountedObject from which to copy memory
     * @param <P> the type of the referenceCountedObject
     * @return this
     */
    public <P extends RCObject> P put(RCObject p) {
        @SuppressWarnings("unchecked")
        P p2 = (P) this;
        return p2;
    }

    /**
     * Calls in effect {@code memset(address + position, b, length)}, where {@code length = sizeof()
     * * (limit - position)}. If limit == 0, it uses position + 1 instead. The way the methods were
     * designed allows constructs such as {@code this.position(0).limit(13).fill(42)};
     *
     * @param b the byte value to fill the memory with
     * @param <P> the type of the referenceCountedObject
     * @return this
     */
    public <P extends RCObject> P fill(int b) {
        @SuppressWarnings("unchecked")
        P p = (P) this;
        return p;
    }

    /**
     * Returns {@code fill(0)}.
     *
     * @param <P> the type of the referenceCountedObject
     * @return this
     */
    public <P extends RCObject> P zero() {
        // repair warning: [unchecked] unchecked cast
        @SuppressWarnings("unchecked")
        P p = (P) this.fill(0);
        return p;
    }

    /**
     * Returns whether the resource is null.
     *
     * @return whether the resource is null
     */
    public boolean isNull() {
        throw new UnsupportedOperationException("Not implemented.");
    }
}
