/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.ndarray.gc;

/** {@code GCSwitch} acts as a switch to put garbage collection on or off. */
public final class SwitchGarbageCollection {

    private static boolean useGarbageCollection;

    /** Hide the constructor of this utility class. */
    private SwitchGarbageCollection() {}

    /**
     * Returns whether to use garbage collection to manage temporary resources.
     *
     * @return the useGarbageCollection
     */
    public static boolean isUseGarbageCollection() {
        return useGarbageCollection;
    }

    /** Switches the garbage collection on. */
    public static void on() {
        useGarbageCollection = true;
    }
}
