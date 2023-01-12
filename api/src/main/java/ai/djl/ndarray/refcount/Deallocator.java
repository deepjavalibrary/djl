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

/**
 * The interface to implement to produce a Deallocator usable by referenceCountedObject.
 *
 * <p>This interface has been derived from {@code org.bytedeco.javacpp.Pointer} by Samuel Audet
 */
public interface Deallocator {
    /** The method to implement to produce a Deallocator usable by referenceCountedObject. */
    void deallocate();
}
