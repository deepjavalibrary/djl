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

/** RCConfig holds configurable parameters. */
public final class RCConfig {

    private static boolean verboseIfResourceAlreadyClosed;

    private RCConfig() {}

    /**
     * If true, a verbose message is printed if a resource is already closed.
     *
     * @return true if the verboseIfResourceAlreadyClosed is set
     */
    public static boolean isVerboseIfResourceAlreadyClosed() {
        return verboseIfResourceAlreadyClosed;
    }

    /**
     * If true, a verbose message is printed if a resource is already closed.
     *
     * @param verboseIfResourceAlreadyClosed parameter to set
     */
    public static void setVerboseIfResourceAlreadyClosed(boolean verboseIfResourceAlreadyClosed) {
        RCConfig.verboseIfResourceAlreadyClosed = verboseIfResourceAlreadyClosed;
    }
}
