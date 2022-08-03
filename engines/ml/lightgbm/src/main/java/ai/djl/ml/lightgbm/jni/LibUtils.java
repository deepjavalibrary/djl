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
package ai.djl.ml.lightgbm.jni;

import ai.djl.engine.EngineException;
import ai.djl.util.ClassLoaderUtils;
import ai.djl.util.Platform;
import ai.djl.util.Utils;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/** Utilities for the {@link ai.djl.ml.lightgbm.LgbmEngine} to load the native binary. */
public final class LibUtils {

    private LibUtils() {}

    /**
     * Loads the native binary for LightGBM.
     *
     * @throws IOException if it fails to download the native library
     */
    public static synchronized void loadNative() throws IOException {
        Platform platform = Platform.fromSystem("lightgbm");

        if (!"x86_64".equals(platform.getOsArch())) {
            throw new IllegalStateException("Only x86 is supported");
        }

        if ("linux".equals(platform.getOsPrefix())) {
            loadNative("linux/x86_64/lib_lightgbm.so", "lib_lightgbm.so");
            loadNative("linux/x86_64/lib_lightgbm_swig.so", "lib_lightgbm_swig.so");
            return;
        }
        if ("osx".equals(platform.getOsPrefix())) {
            loadNative("osx/x86_64/lib_lightgbm.dylib", "lib_lightgbm.dylib");
            loadNative("osx/x86_64/lib_lightgbm_swig.dylib", "lib_lightgbm_swig.dylib");
            return;
        }
        if ("win".equals(platform.getOsPrefix())) {
            loadNative("windows/x86_64/lib_lightgbm.dll", "lib_lightgbm.dll");
            loadNative("windows/x86_64/lib_lightgbm_swig.dll", "lib_lightgbm_swig.dll");
            return;
        }

        throw new IllegalStateException("No LightGBM Engine matches your platform");
    }

    private static void loadNative(String resourcePath, String name) throws IOException {
        Path cacheFolder = Utils.getEngineCacheDir("lightgbm");
        Path libFile = cacheFolder.resolve(name);
        if (!libFile.toFile().exists()) {

            if (!cacheFolder.toFile().exists()) {
                Files.createDirectories(cacheFolder);
            }

            resourcePath = "com/microsoft/ml/lightgbm/" + resourcePath;
            Path tmp = Files.createTempDirectory("lightgbm-" + name).resolve(name);
            try (InputStream is = ClassLoaderUtils.getResourceAsStream(resourcePath)) {
                Files.copy(is, tmp, StandardCopyOption.REPLACE_EXISTING);
            }
            Utils.moveQuietly(tmp, libFile);
        }
        try {
            System.load(libFile.toString());
        } catch (UnsatisfiedLinkError err) {
            throw new EngineException("Cannot load library: " + name, err);
        }
    }
}
