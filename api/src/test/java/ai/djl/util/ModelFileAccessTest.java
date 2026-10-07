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
package ai.djl.util;

import ai.djl.ModelException;
import ai.djl.ndarray.NDList;
import ai.djl.nn.LambdaBlock;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;

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
import java.util.List;

/**
 * Tests for resolving files named by model arguments in the model directory, in {@link
 * Utils#resolveModelFile} and {@link ai.djl.BaseModel#getArtifact(String)}.
 */
public class ModelFileAccessTest {

    private String savedAllowOutside;
    private Path root;
    private Path modelDir;

    @BeforeMethod
    public void setUp() throws IOException {
        // The build forwards every ai.djl.* system property into the test JVM, so record the flag
        // and put it back in cleanup() rather than leaving it cleared for later tests.
        savedAllowOutside = System.getProperty("ai.djl.allow_files_outside_model_dir");
        if (Utils.getenv("DJL_ALLOW_FILES_OUTSIDE_MODEL_DIR") != null) {
            throw new SkipException("DJL_ALLOW_FILES_OUTSIDE_MODEL_DIR is set in the environment");
        }
        System.clearProperty("ai.djl.allow_files_outside_model_dir");

        root = Files.createTempDirectory("djl-model-files");
        modelDir = Files.createDirectories(root.resolve("model"));
        Files.write(modelDir.resolve("synset.txt"), "cat\ndog".getBytes(StandardCharsets.UTF_8));
        Files.write(root.resolve("outside.txt"), "outside".getBytes(StandardCharsets.UTF_8));
        // Shares the "model" prefix, so a check that compared strings would treat it as inside.
        Path sibling = Files.createDirectories(root.resolve("model-other"));
        Files.write(sibling.resolve("file.txt"), "sibling".getBytes(StandardCharsets.UTF_8));
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

    // ----- Utils.resolveModelFile -----

    @Test
    public void testFileInsideModelDir() {
        Path file = modelDir.resolve("synset.txt");
        Assert.assertEquals(Utils.resolveModelFile(modelDir, "synset.txt"), file);
        // An absolute path is accepted when it points into the model directory.
        String absolute = file.toAbsolutePath().toString();
        Assert.assertEquals(Utils.resolveModelFile(modelDir, absolute), file.toAbsolutePath());
        // ".." is fine as long as the result stays inside.
        Assert.assertEquals(
                Utils.resolveModelFile(modelDir, "sub/../synset.txt"),
                modelDir.resolve("sub/../synset.txt"));
    }

    @Test
    public void testFileOutsideModelDirIsRejected() {
        String[] names = {
            "../outside.txt",
            "sub/../../outside.txt",
            root.resolve("outside.txt").toAbsolutePath().toString(),
            "../model-other/file.txt",
            root.resolve("model-other/file.txt").toAbsolutePath().toString()
        };
        for (String name : names) {
            IllegalArgumentException e =
                    Assert.expectThrows(
                            IllegalArgumentException.class,
                            () -> Utils.resolveModelFile(modelDir, name));
            Assert.assertTrue(
                    e.getMessage().contains("outside the model directory"),
                    name + " -> " + e.getMessage());
        }
    }

    @Test
    public void testFileOutsideModelDirAllowedWithOptIn() {
        Assert.assertFalse(Utils.isFileOutsideModelDirAllowed());
        System.setProperty("ai.djl.allow_files_outside_model_dir", "true");
        Assert.assertTrue(Utils.isFileOutsideModelDirAllowed());
        Assert.assertEquals(
                Utils.resolveModelFile(modelDir, "../outside.txt"),
                modelDir.resolve("../outside.txt"));
    }

    // ----- Model artifacts: BaseModel.getArtifact -----

    @Test
    public void testGetArtifact() throws IOException, ModelException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelPath(modelDir)
                        .optBlock(new LambdaBlock(a -> a, "model"))
                        .optEngine("PyTorch")
                        .optOption("hasParameter", "false")
                        .build();
        try (ZooModel<NDList, NDList> model = criteria.loadModel()) {
            List<String> synset = model.getArtifact("synset.txt", Utils::readLines);
            Assert.assertEquals(synset, Arrays.asList("cat", "dog"));

            Assert.assertThrows(
                    IllegalArgumentException.class, () -> model.getArtifact("../outside.txt"));
            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> model.getArtifact("../outside.txt", Utils::readLines));
            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> model.getArtifactAsStream("../outside.txt"));

            System.setProperty("ai.djl.allow_files_outside_model_dir", "true");
            List<String> outside = model.getArtifact("../outside.txt", Utils::readLines);
            Assert.assertEquals(outside, Collections.singletonList("outside"));
        }
    }
}
