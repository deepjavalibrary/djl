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
package ai.djl.translate;

import ai.djl.ModelException;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.nn.Blocks;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Tests how {@link ServingTranslatorFactory} resolves a translator that a model bundles, which
 * depends on whether loading classes shipped with a model is enabled.
 */
public class BundledTranslatorGuardTest {

    @Test
    public void testBundledTranslatorIsNotSilentlyReplaced()
            throws IOException, TranslateException, ModelException {
        // The environment variable takes precedence over the system property, so an environment
        // that sets it would make the property below a no-op.
        if (Utils.getenv("DJL_LOAD_BUNDLED_CLASSES") != null) {
            throw new SkipException("DJL_LOAD_BUNDLED_CLASSES is set in the environment");
        }
        String saved = System.getProperty("ai.djl.load_bundled_classes");
        System.clearProperty("ai.djl.load_bundled_classes");
        Path path = Paths.get("build/bundledTranslatorGuard");
        Utils.deleteQuietly(path);
        Path classes = path.resolve("libs").resolve("classes");
        Files.createDirectories(classes);
        // Stands in for a translator the model ships. The bytes are never read: with bundled
        // loading off the file is not opened at all, and the control below only needs the lookup to
        // find no usable translator in it.
        Files.write(
                classes.resolve("Bundled.class"),
                new byte[] {(byte) 0xCA, (byte) 0xFE, (byte) 0xBA, (byte) 0xBE});
        // Utils.getNestedModelDir descends into a lone subdirectory, so a model directory holding
        // only libs/ resolves to libs/ itself and is never seen as bundling anything. A real model
        // has files beside libs/; this stands in for them. Without it these tests pass vacuously.
        Files.write(path.resolve("serving.properties"), new byte[0]);
        try {
            // The model bundles a translator and names none, so the name could only have come from
            // that bundled content. Loading with a substituted default translator would serve the
            // model with different pre- and post-processing while looking like a correct load, so
            // the load has to fail instead.
            ModelNotFoundException e =
                    Assert.expectThrows(
                            ModelNotFoundException.class, () -> bundledCriteria(path).loadModel());
            // Criteria wraps the loader's failure, which in turn wraps the TranslateException, so
            // assert over the whole cause chain rather than a fixed depth.
            String chain = causeChain(e);
            Assert.assertTrue(
                    chain.contains(TranslateException.class.getName()),
                    "expected a translator failure in the chain: " + chain);
            Assert.assertTrue(
                    chain.contains("DJL_LOAD_BUNDLED_CLASSES"),
                    "the failure must name the flag: " + chain);

            // Control: with the opt-in the bundled content is consulted. It holds no usable
            // translator, so the default is used and the load succeeds. That shows the failure
            // above comes from the guard, not from anything else about the model directory.
            System.setProperty("ai.djl.load_bundled_classes", "true");
            try (ZooModel<Input, Output> model = bundledCriteria(path).loadModel()) {
                Assert.assertEquals(
                        model.getTranslator().getClass().getSimpleName(), "ImageServingTranslator");
            }
        } finally {
            if (saved == null) {
                System.clearProperty("ai.djl.load_bundled_classes");
            } else {
                System.setProperty("ai.djl.load_bundled_classes", saved);
            }
            Utils.deleteQuietly(path);
        }
    }

    @Test
    public void testModelWithoutBundledClassesStillFallsBack()
            throws IOException, TranslateException, ModelException {
        if (Utils.getenv("DJL_LOAD_BUNDLED_CLASSES") != null) {
            throw new SkipException("DJL_LOAD_BUNDLED_CLASSES is set in the environment");
        }
        System.clearProperty("ai.djl.load_bundled_classes");
        Path path = Paths.get("build/noBundledTranslatorGuard");
        Utils.deleteQuietly(path);
        // A libs directory holding no class content is not a bundled translator, so the ordinary
        // fallback to a default translator must be unaffected. The extra file keeps
        // getNestedModelDir from descending into libs/, which would bypass the check entirely and
        // let this pass for the wrong reason.
        Files.createDirectories(path.resolve("libs"));
        Files.write(path.resolve("serving.properties"), new byte[0]);
        try (ZooModel<Input, Output> model = bundledCriteria(path).loadModel()) {
            Assert.assertEquals(
                    model.getTranslator().getClass().getSimpleName(), "ImageServingTranslator");
        } finally {
            Utils.deleteQuietly(path);
        }
    }

    private static String causeChain(Throwable t) {
        StringBuilder sb = new StringBuilder();
        for (Throwable c = t; c != null; c = c.getCause()) {
            sb.append(c.getClass().getName()).append(": ").append(c.getMessage()).append('\n');
        }
        return sb.toString();
    }

    private static Criteria<Input, Output> bundledCriteria(Path path) {
        return Criteria.builder()
                .setTypes(Input.class, Output.class)
                .optModelPath(path)
                .optModelName("identity")
                .optArgument("application", "image_classification")
                .optOption("hasParameter", "false")
                .optBlock(Blocks.identityBlock())
                .build();
    }
}
