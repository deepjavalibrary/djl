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

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.Iterator;

/**
 * {@link RCObject} objects attach themselves automatically on creation to the first {@link RCScope}
 * found in {@link #SCOPE_STACK} that they can to based on the classes found in {@link #forClasses}.
 * The user can then call {@link #deallocate()}, or rely on {@link #close()} to release in a timely
 * fashion all attached referenceCountedObject objects, instead of relying on the garbage collector.
 *
 * <p>This class has been derived from {@code org.bytedeco.javacpp.PointerScope} by Samuel Audet
 */
public class RCScope implements AutoCloseable {
    /**
     * A thread-local stack of {@link RCScope} objects. referenceCountedObject objects attach
     * themselves automatically to the first one they can to on the stack.
     */
    static final ThreadLocal<Deque<RCScope>> SCOPE_STACK =
            new ThreadLocal<Deque<RCScope>>() {
                @Override
                protected Deque<RCScope> initialValue() {
                    return new ArrayDeque<RCScope>();
                }
            };

    private static final Logger logger = LoggerFactory.getLogger(RCScope.class);
    /** The stack keeping references to attached {@link RCObject} objects. */
    Deque<RCObject> referenceCountedObjectStack = new ArrayDeque<>();
    /** When not empty, indicates the classes of objects that are allowed to be attached. */
    Class<? extends RCObject>[] forClasses;
    /**
     * When set to true, the next call to {@link #close()} does not release but resets this
     * variable.
     */
    boolean extend;

    /**
     * Creates a new scope accepting all referenceCountedObject types and pushes itself on the
     * {@link #SCOPE_STACK}.
     */
    public RCScope() {
        this((Class<? extends RCObject>[]) null);
    }

    /**
     * Initializes {@link #forClasses}, and adds itself as first (push) on the {@link #SCOPE_STACK}.
     *
     * @param forClasses the classes of objects that are allowed to be attached
     */
    @SafeVarargs
    @SuppressWarnings("varargs")
    public RCScope(Class<? extends RCObject>... forClasses) {
        if (logger.isDebugEnabled()) {
            logger.debug("Opening " + this);
        }
        this.forClasses = forClasses;
        SCOPE_STACK.get().addFirst(this);
    }

    /**
     * Returns {@code SCOPE_STACK.get().peekFirst()} (peek), the last opened scope not yet closed.
     *
     * @return the last opened scope not yet closed
     */
    public static RCScope getInnerScope() {
        return SCOPE_STACK.get().peekFirst();
    }

    /**
     * Returns {@code SCOPE_STACK.get().iterator()}, all scopes not yet closed.
     *
     * @return all scopes not yet closed
     */
    public static Iterator<RCScope> getScopeIterator() {
        return SCOPE_STACK.get().iterator();
    }

    /**
     * When not empty, returns the classes of objects that are allowed to be attached.
     *
     * @return the classes of objects that are allowed to be attached
     */
    public Class<? extends RCObject>[] forClasses() {
        return forClasses;
    }

    /**
     * Pushes the referenceCountedObject onto the {@link #referenceCountedObjectStack} of this Scope
     * and calls {@link RCObject#retainReference()}.
     *
     * @param p the referenceCountedObject to attach
     * @return the referenceCountedObject
     * @throws IllegalArgumentException when it is not an instance of a class in {@link
     *     #forClasses}.
     */
    public RCScope attach(RCObject p) {
        if (logger.isDebugEnabled()) {
            logger.debug("Attaching " + p + " to " + this);
        }
        if (forClasses != null && forClasses.length > 0) {
            boolean found = false;
            for (Class<? extends RCObject> c : forClasses) {
                if (c != null && c.isInstance(p)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw new IllegalArgumentException(
                        p
                                + " is not an instance of a class in forClasses: "
                                + Arrays.toString(forClasses));
            }
        }
        referenceCountedObjectStack.push(p);
        p.retainReference();
        return this;
    }

    /**
     * Removes the referenceCountedObject from the {@link #referenceCountedObjectStack} of this
     * Scope and calls {@link RCObject#releaseReference()}.
     *
     * @param p the referenceCountedObject to detach
     * @return the referenceCountedObject
     */
    public RCScope detach(RCObject p) {
        if (logger.isDebugEnabled()) {
            logger.debug("Detaching " + p + " from " + this);
        }
        referenceCountedObjectStack.remove(p);
        p.releaseReference();
        return this;
    }

    /**
     * Extends the life of this scope past the next call to {@link #close()} by setting the {@link
     * #extend} flag.
     *
     * @return this scope
     */
    public RCScope extend() {
        if (logger.isDebugEnabled()) {
            logger.debug("Extending " + this);
        }
        extend = true;
        return this;
    }

    /**
     * Pops from {@link #referenceCountedObjectStack} all attached ReferenceCountedObjects, calls
     * {@link RCObject#releaseReference()} on them, unless extended, in which case it only resets
     * the {@link #extend} flag instead, and finally removes itself from {@link #SCOPE_STACK}.
     */
    @Override
    public void close() {
        if (logger.isDebugEnabled()) {
            logger.debug("Closing " + this);
        }
        if (extend) {
            extend = false;
        } else {
            while (referenceCountedObjectStack.size() > 0) {
                referenceCountedObjectStack.pop().releaseReference();
            }
        }
        SCOPE_STACK.get().remove(this);
    }

    /**
     * Pops from {@link #referenceCountedObjectStack} all attached ReferenceCountedObjects, and
     * calls {@link RCObject#deallocate()} on them.
     */
    public void deallocate() {
        if (logger.isDebugEnabled()) {
            logger.debug("Deallocating " + this);
        }
        while (referenceCountedObjectStack.size() > 0) {
            referenceCountedObjectStack.pop().deallocate();
        }
    }

    /**
     * A method that does nothing. You may use it if you do not have a better way to suppress the
     * warning of a created but not explicitly used scope.
     */
    public void suppressNotUsedWarning() {
        // do nothing
    }
}
