/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.onnxruntime.engine;

import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Covers the two option gates without a session: both run before ONNX Runtime is reached, and the
 * session they are reached through needs the native library and a real model file.
 */
public class OrtOptionGateTest {

    private static final String OUTSIDE_PROPERTY = "ai.djl.allow_files_outside_model_dir";
    private static final String CUSTOM_OP_PROPERTY = "ai.djl.allow_custom_op_library";

    private String savedOutside;
    private String savedCustomOp;

    @BeforeMethod
    public void saveProperties() {
        // Both flags read the environment variable first, so an environment that sets either would
        // make the matching property below a no-op.
        if (Utils.getenv("DJL_ALLOW_FILES_OUTSIDE_MODEL_DIR") != null
                || Utils.getenv("DJL_ALLOW_CUSTOM_OP_LIBRARY") != null) {
            throw new SkipException("A DJL_ALLOW_* flag is set in the environment");
        }
        // The build forwards every ai.djl.* system property into the test JVM, so these may
        // already be set by the developer running the suite. Record them, clear them so the
        // assertions below see the default, and put them back in cleanup().
        savedOutside = System.getProperty(OUTSIDE_PROPERTY);
        savedCustomOp = System.getProperty(CUSTOM_OP_PROPERTY);
        System.clearProperty(OUTSIDE_PROPERTY);
        System.clearProperty(CUSTOM_OP_PROPERTY);
    }

    @AfterMethod
    public void cleanup() {
        restore(OUTSIDE_PROPERTY, savedOutside);
        restore(CUSTOM_OP_PROPERTY, savedCustomOp);
    }

    private static void restore(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }

    @Test
    public void testProfilerPrefixResolvesInsideModelDir() throws IOException {
        Path modelDir = Files.createTempDirectory("ort-profiler");
        Assert.assertEquals(
                OrtModel.resolveProfilerPrefix(modelDir, "profile"),
                modelDir.resolve("profile").toString());

        // A nested prefix keeps working, and the directory it names is created so ONNX Runtime can
        // write the timestamped file into it.
        Assert.assertEquals(
                OrtModel.resolveProfilerPrefix(modelDir, "logs/profile"),
                modelDir.resolve("logs/profile").toString());
        Assert.assertTrue(Files.isDirectory(modelDir.resolve("logs")));
    }

    @Test
    public void testProfilerPrefixEscapingModelDirIsRefused() throws IOException {
        Path modelDir = Files.createTempDirectory("ort-profiler");
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> OrtModel.resolveProfilerPrefix(modelDir, "../profile"));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> OrtModel.resolveProfilerPrefix(modelDir, "/tmp/profile"));

        // A prefix that is the model directory itself is refused as well: ONNX Runtime appends the
        // timestamp to it, which lands on a sibling of the directory rather than inside it.
        Assert.assertThrows(
                IllegalArgumentException.class, () -> OrtModel.resolveProfilerPrefix(modelDir, ""));
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> OrtModel.resolveProfilerPrefix(modelDir, "."));

        // The documented override still applies, so an application that wrote its profiles
        // elsewhere keeps working.
        System.setProperty(OUTSIDE_PROPERTY, "true");
        Assert.assertEquals(
                OrtModel.resolveProfilerPrefix(modelDir, "../profile"),
                modelDir.resolve("../profile").toString());
    }

    @Test
    public void testProfilerPrefixThroughLinkIsRefused() throws IOException {
        // Lexically inside the model directory, but the link sends the write elsewhere.
        Path modelDir = Files.createTempDirectory("ort-profiler");
        Path outside = Files.createTempDirectory("ort-outside");
        Files.createSymbolicLink(modelDir.resolve("logs"), outside);
        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> OrtModel.resolveProfilerPrefix(modelDir, "logs/sub/profile"));
        Assert.assertFalse(Files.exists(outside.resolve("sub")));
    }

    @Test
    public void testCustomOpLibraryIsOptIn() {
        Assert.assertNull(OrtModel.allowedCustomOpLibrary("/tmp/custom.so"));
        // Null stays null, so the caller falls back to the classpath library either way.
        Assert.assertNull(OrtModel.allowedCustomOpLibrary(null));

        System.setProperty(CUSTOM_OP_PROPERTY, "true");
        Assert.assertEquals(OrtModel.allowedCustomOpLibrary("/tmp/custom.so"), "/tmp/custom.so");

        // Anything other than true leaves it disabled rather than enabling it by accident.
        System.setProperty(CUSTOM_OP_PROPERTY, "1");
        Assert.assertNull(OrtModel.allowedCustomOpLibrary("/tmp/custom.so"));
    }
}
