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
package ai.djl.paddlepaddle.jna;

import com.sun.jna.Native;
import java.io.File;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utilities for finding the Paddle Engine binary on the System.
 *
 * <p>The Engine will be searched for in a variety of locations in the following order:
 *
 * <ol>
 *   <li>In the path specified by the Paddle_LIBRARY_PATH environment variable
 *   <li>In a jar file location in the classpath. These jars can be created with the paddle-native
 *       module.
 * </ol>
 */
@SuppressWarnings("MissingJavadocMethod")
public final class LibUtils {

    private static final Logger logger = LoggerFactory.getLogger(LibUtils.class);

    private static final String LIB_NAME = "paddle_fluid";

    private LibUtils() {}

    public static PaddleLibrary loadLibrary() {
        String libName = getLibName();
        logger.debug("Loading paddle library from: {}", libName);

        return Native.load(libName, PaddleLibrary.class);
    }

    public static String getLibName() {
        String libName = LibUtils.findOverrideLibrary();
        if (libName == null) {
            // libName = LibUtils.findLibraryInClasspath();
            if (libName == null) {
                libName = LIB_NAME;
            }
        }
        return libName;
    }

    private static String findOverrideLibrary() {
        String libPath = System.getenv("PADDLE_LIBRARY_PATH");
        if (libPath != null) {
            String libName = findLibraryInPath(libPath);
            if (libName != null) {
                return libName;
            }
        }

        libPath = System.getProperty("java.library.path");
        if (libPath != null) {
            return findLibraryInPath(libPath);
        }
        return null;
    }

    private static String findLibraryInPath(String libPath) {
        String[] paths = libPath.split(File.pathSeparator);
        String mappedLibNames = System.mapLibraryName(LIB_NAME);

        for (String path : paths) {
            File p = new File(path);
            if (!p.exists()) {
                continue;
            }
            if (p.isFile() && p.getName().endsWith(mappedLibNames)) {
                return p.getAbsolutePath();
            }

            File file = new File(path, mappedLibNames);
            if (file.exists() && file.isFile()) {
                return file.getAbsolutePath();
            }
        }
        return null;
    }
}
