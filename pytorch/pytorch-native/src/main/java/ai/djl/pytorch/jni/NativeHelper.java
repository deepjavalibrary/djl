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
package ai.djl.pytorch.jni;

/** A helper class allows engine shared library to be loaded from different class loader. */
public final class NativeHelper {

    private NativeHelper() {}

    /**
     * Load native shared library from file.
     *
     * @param path the file to load
     */
    public static void load(String path) {
        System.load(path); // NOPMD
    }
}
