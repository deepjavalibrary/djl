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
package ai.djl.dlr.jni;

/**
 * Utilities for finding the DLR Engine binary on the System.
 *
 * <p>The Engine will be searched for in a variety of locations in the following order:
 *
 * <ol>
 *   <li>In the path specified by the DLR_LIBRARY_PATH environment variable
 * </ol>
 */
@SuppressWarnings("MissingJavadocMethod")
public final class LibUtils {

    private LibUtils() {}

    public static void loadLibrary() {
        // TODO implement
        System.load("/Users/leecheng/workspace/djl/dlr/dlr-native/libdlr.dylib");
        System.load("/Users/leecheng/.djl.ai/dlr/1.5.0-cpu-osx-x86_64/libdjl_dlr.dylib");
    }
}
