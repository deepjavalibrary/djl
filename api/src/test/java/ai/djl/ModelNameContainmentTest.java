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
package ai.djl;

import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

/** Tests that a model name only selects files inside the model directory. */
public class ModelNameContainmentTest {

    private String savedAllowOutside;
    private Path root;
    private Path modelDir;
    private String[] outsideNames;

    @BeforeMethod
    public void setUp() throws IOException {
        // The build forwards every ai.djl.* system property into the test JVM, so record the flag
        // and put it back in cleanup() rather than leaving it cleared for later tests.
        savedAllowOutside = System.getProperty("ai.djl.allow_files_outside_model_dir");
        if (Utils.getenv("DJL_ALLOW_FILES_OUTSIDE_MODEL_DIR") != null) {
            throw new SkipException("DJL_ALLOW_FILES_OUTSIDE_MODEL_DIR is set in the environment");
        }
        System.clearProperty("ai.djl.allow_files_outside_model_dir");

        root = Files.createTempDirectory("djl-model-name");
        modelDir = Files.createDirectories(root.resolve("model"));
        // Files the names below would select if they were not contained.
        Files.write(root.resolve("outside.pt"), "not a model".getBytes(StandardCharsets.UTF_8));
        Files.write(
                root.resolve("outside-0000.params"), "not params".getBytes(StandardCharsets.UTF_8));
        outsideNames =
                new String[] {"../outside", root.resolve("outside").toAbsolutePath().toString()};
    }

    @AfterMethod
    public void cleanup() {
        if (savedAllowOutside == null) {
            System.clearProperty("ai.djl.allow_files_outside_model_dir");
        } else {
            System.setProperty("ai.djl.allow_files_outside_model_dir", savedAllowOutside);
        }
        if (root != null) {
            Utils.deleteQuietly(root);
        }
    }

    @Test
    public void testModelNameOutsideModelDirIsRejected() {
        for (String name : outsideNames) {
            // The name a model artifact supplies through its options.
            Criteria<NDList, NDList> fromOptions =
                    Criteria.builder()
                            .setTypes(NDList.class, NDList.class)
                            .optModelPath(modelDir)
                            .optEngine("PyTorch")
                            .optOption("modelName", name)
                            .build();
            assertRejected(fromOptions, name);

            // The same name passed by the application.
            Criteria<NDList, NDList> fromCriteria =
                    Criteria.builder()
                            .setTypes(NDList.class, NDList.class)
                            .optModelPath(modelDir)
                            .optEngine("PyTorch")
                            .optModelName(name)
                            .build();
            assertRejected(fromCriteria, name);
        }
    }

    @Test
    public void testParameterFileOutsideModelDirIsRejected() throws IOException {
        try (Model model = Model.newInstance("model", "PyTorch")) {
            BaseModel base = (BaseModel) model;
            base.setModelDir(modelDir);
            Map<String, String> epoch = Collections.singletonMap("epoch", "0");
            Map<String, String> noEpoch = Collections.emptyMap();
            Assert.assertEquals(
                    base.paramPathResolver("model", epoch),
                    modelDir.toAbsolutePath().resolve("model-0000.params"));
            // With no epoch and no parameter file there is nothing to load, which is not an error.
            Assert.assertNull(base.paramPathResolver("model", noEpoch));
            for (String name : outsideNames) {
                for (Map<String, String> options : Arrays.asList(epoch, noEpoch)) {
                    IllegalArgumentException e =
                            Assert.expectThrows(
                                    IllegalArgumentException.class,
                                    () -> base.paramPathResolver(name, options));
                    Assert.assertTrue(
                            e.getMessage().contains("outside the model directory"),
                            name + " " + options + " -> " + e.getMessage());
                }
            }
        }
    }

    private static void assertRejected(Criteria<NDList, NDList> criteria, String name) {
        Throwable e = Assert.expectThrows(Exception.class, criteria::loadModel);
        for (Throwable t = e; t != null; t = t.getCause()) {
            if (t instanceof IllegalArgumentException
                    && t.getMessage() != null
                    && t.getMessage().contains("outside the model directory")) {
                return;
            }
        }
        Assert.fail(name + " was not rejected as outside the model directory", e);
    }
}
