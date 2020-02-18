/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.tensorflow.engine;

import ai.djl.engine.EngineException;
import ai.djl.util.Utils;
import ai.djl.util.cuda.CudaUtils;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@SuppressWarnings("MissingJavadocMethod")
public final class LibUtils {
    private static final Logger logger = LoggerFactory.getLogger(LibUtils.class);

    private static String extension;

    private LibUtils() {}

    public static String loadLibrary() {
        String libName = getLibName();
        logger.debug("Loading TensorFlow library from: {}", libName);
        System.load(libName + "/libtensorflow_jni" + extension);
        System.load(libName + "/libtensorflow_framework" + extension);
        return libName;
    }

    public static String getLibName() {
        String libName;
        try {
            libName = LibUtils.getTensorFlowLib();
        } catch (IOException | InterruptedException e) {
            throw new IllegalStateException("Failed to find TensorFlow library", e);
        }
        return libName;
    }

    private static String getTensorFlowLib() throws IOException, InterruptedException {
        boolean useGPU = false;
        if (CudaUtils.getGpuCount() > 0) {
            useGPU = true;
        }
        String osName = System.getProperty("os.name");
        if (osName.startsWith("Mac")) {
            extension = ".dylib";
            return downloadTensorFlow("libtensorflow_jni-cpu-darwin-x86_64");
        } else if (osName.startsWith("Linux")) {
            extension = ".so";
            if (useGPU) {
                return downloadTensorFlow("libtensorflow_jni-gpu-linux-x86_64");
            } else {
                return downloadTensorFlow("libtensorflow_jni-cpu-linux-x86_64");
            }
        } else if (osName.startsWith("Win")) {
            throw new EngineException(
                    "TensorFlow engine does not support Windows yet," + "please use MXNet engine");
        } else {
            throw new EngineException("OS not supported: " + osName);
        }
    }

    private static String downloadTensorFlow(String fileName) {
        String userHome = System.getProperty("user.home");
        Path dir = Paths.get(userHome, ".tensorflow/cache/");
        Path path = dir.resolve(fileName);
        if (Files.exists(path)) {
            return path.toAbsolutePath().toString();
        }

        Path tmp = Paths.get(userHome, ".tensorflow/cache/tmp");

        String link =
                "https://storage.googleapis.com/tensorflow-nightly/github/tensorflow/lib_package/";
        try (InputStream is = new URL(link + fileName + ".tar.gz").openStream()) {
            Files.createDirectories(tmp);
            Path tarFile = tmp.resolve(fileName + ".tar.gz");
            Files.copy(is, tarFile);
            Files.createDirectories(path);
            Runtime.getRuntime()
                    .exec("tar -xzvf " + tarFile.toAbsolutePath() + " -C " + path.toAbsolutePath())
                    .waitFor();
            return path.toAbsolutePath().toString();
        } catch (IOException | InterruptedException e) {
            throw new IllegalStateException("Failed to download TensorFlow library", e);
        } finally {
            Utils.deleteQuietly(tmp);
        }
    }
}
