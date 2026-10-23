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
import ai.djl.util.ClassLoaderUtils;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Map;
import java.util.function.Consumer;

public class ModelClassAllowlistTest {

    private static final String PROPERTY = "ai.djl.allowed_model_classes";
    private static final String OTHER_PACKAGE = "com.example.allowed.*";

    /** A factory DJL also names itself, through {@code detectTranslatorFactory}. */
    private static final String IMAGE_FACTORY =
            "ai.djl.modality.cv.translator.ImageClassificationTranslatorFactory";

    private String savedAllowlist;

    @BeforeMethod
    public void saveAllowlist() {
        // The environment variable takes precedence over the system property, so an environment
        // that sets it would make every property below a no-op.
        if (Utils.getenv("DJL_ALLOWED_MODEL_CLASSES") != null) {
            throw new SkipException("DJL_ALLOWED_MODEL_CLASSES is set in the environment");
        }
        // The build forwards every ai.djl.* system property into the test JVM, so this may already
        // be set by the developer running the suite. Record it, clear it so each test sets its own
        // allowlist, and put it back in cleanup().
        savedAllowlist = System.getProperty(PROPERTY);
        System.clearProperty(PROPERTY);
    }

    @AfterMethod
    public void cleanup() {
        if (savedAllowlist == null) {
            System.clearProperty(PROPERTY);
        } else {
            System.setProperty(PROPERTY, savedAllowlist);
        }
    }

    @Test
    public void testAllowlistAppliesToModelNamedClassesOnly() {
        ClassLoader cl = ClassLoaderUtils.getContextClassLoader();
        System.setProperty(PROPERTY, OTHER_PACKAGE);

        // A name DJL supplies itself keeps resolving: detectTranslatorFactory and
        // NoopServingTranslator's CsvTranslator go through initClass, which is not allowlisted.
        Assert.assertNotNull(
                ClassLoaderUtils.initClass(cl, TranslatorFactory.class, IMAGE_FACTORY));

        // The same name supplied by a model is refused.
        Assert.assertNull(
                ClassLoaderUtils.initModelClass(cl, TranslatorFactory.class, IMAGE_FACTORY));
        // ... including through findImplementation, which every caller feeds from model arguments.
        Assert.assertNull(
                ClassLoaderUtils.findImplementation(
                        Paths.get("build"), TranslatorFactory.class, IMAGE_FACTORY));

        // Negative control: both model-named lookups succeed once the allowlist admits the class,
        // so the assertions above are the allowlist and not a class that cannot load.
        System.setProperty(PROPERTY, OTHER_PACKAGE + ",ai.djl.modality.cv.*");
        Assert.assertNotNull(
                ClassLoaderUtils.initModelClass(cl, TranslatorFactory.class, IMAGE_FACTORY));
        Assert.assertNotNull(
                ClassLoaderUtils.findImplementation(
                        Paths.get("build"), TranslatorFactory.class, IMAGE_FACTORY));

        // And with no allowlist configured, nothing is restricted.
        System.clearProperty(PROPERTY);
        Assert.assertNotNull(
                ClassLoaderUtils.initModelClass(cl, TranslatorFactory.class, IMAGE_FACTORY));
    }

    @Test
    public void testDeferredFactoryRefusesAClassTheAllowlistExcludes() throws TranslateException {
        String noop = "ai.djl.translate.NoopServingTranslatorFactory";
        DeferredTranslatorFactory factory = new DeferredTranslatorFactory();
        Map<String, String> arguments = Collections.singletonMap("translatorFactory", noop);

        System.setProperty(PROPERTY, OTHER_PACKAGE);
        Assert.assertThrows(
                TranslateException.class,
                () -> factory.newInstance(Input.class, Output.class, null, arguments));

        System.setProperty(PROPERTY, "ai.djl.translate.*");
        Assert.assertNotNull(factory.newInstance(Input.class, Output.class, null, arguments));
    }

    @Test
    public void testBatchifierNameIsAllowlisted() {
        String name = "ai.djl.translate.SimplePaddingStackBatchifier";

        System.setProperty(PROPERTY, OTHER_PACKAGE);
        Assert.assertThrows(IllegalArgumentException.class, () -> Batchifier.fromString(name));
        // DJL's own batchifier names short-circuit before the allowlisted branch.
        Assert.assertEquals(Batchifier.fromString("stack"), Batchifier.STACK);

        System.setProperty(PROPERTY, "ai.djl.translate.*");
        Assert.assertEquals(
                Batchifier.fromString(name).getClass(), SimplePaddingStackBatchifier.class);
    }

    @Test
    public void testNullNameIsRefusedRatherThanThrowing() {
        System.setProperty(PROPERTY, OTHER_PACKAGE);
        ClassLoader cl = ClassLoaderUtils.getContextClassLoader();
        Assert.assertNull(ClassLoaderUtils.initModelClass(cl, TranslatorFactory.class, null));
    }

    /**
     * Pins the {@code fromModel} leg of {@link ServingTranslatorFactory}, which is only reachable
     * from a model's {@code serving.properties} — {@code BaseModelLoader} consumes a criteria-level
     * {@code translatorFactory} before that file is read. Needs a working engine.
     */
    @Test
    public void testServingPropertiesFactoryIsAllowlisted()
            throws IOException, ModelException, TranslateException {
        // The fixture's serving.properties names a blockFactory as well, so the allowlist admits
        // ai.djl.nn to let the load reach the translator, and excludes only the factory the model
        // names. Both go through the model-named path.
        System.setProperty(PROPERTY, "ai.djl.nn.*");
        Assert.assertThrows(ModelNotFoundException.class, ModelClassAllowlistTest::loadFixture);

        System.setProperty(PROPERTY, "ai.djl.nn.*,ai.djl.modality.cv.*");
        Assert.assertEquals(loadFixture(), "ImageServingTranslator");
    }

    private static String loadFixture() throws IOException, ModelException, TranslateException {
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        // RpcModel.load only records the model directory, so the gate under test
                        // runs without a native engine.
                        .optEngine("RPC")
                        .optModelPath(Paths.get("src/test/resources/identity"))
                        // The fixture's serving.properties sets this too; set it here so the test
                        // does not depend on that file.
                        .optOption("hasParameter", "false")
                        .build();
        try (ZooModel<Input, Output> model = criteria.loadModel()) {
            return model.getTranslator().getClass().getSimpleName();
        }
    }

    /**
     * Pins {@code BaseModelLoader}'s sink, where the name comes from the criteria or the artifact
     * metadata rather than from {@code serving.properties}. Driven through {@code loadModel} so it
     * observes which helper that code picks, so it needs a working engine.
     */
    @Test
    public void testCriteriaFactoryIsAllowlisted()
            throws IOException, ModelException, TranslateException {
        System.setProperty(PROPERTY, OTHER_PACKAGE);
        Assert.assertThrows(
                ModelNotFoundException.class, () -> loadWithCriteriaFactory(IMAGE_FACTORY));

        System.setProperty(PROPERTY, OTHER_PACKAGE + ",ai.djl.modality.cv.*");
        Assert.assertEquals(loadWithCriteriaFactory(IMAGE_FACTORY), "ImageServingTranslator");
    }

    /** Needs a working engine: the name DJL detects itself must load with an allowlist set. */
    @Test
    public void testDetectedFactoryStillLoadsWithAllowlistSet()
            throws IOException, ModelException, TranslateException {
        System.setProperty(PROPERTY, OTHER_PACKAGE);
        Assert.assertEquals(
                loadScratchModel(c -> c.optArgument("application", "image_classification")),
                "ImageServingTranslator");
    }

    private static String loadWithCriteriaFactory(String factoryClass)
            throws IOException, ModelException, TranslateException {
        return loadScratchModel(c -> c.optArgument("translatorFactory", factoryClass));
    }

    private static String loadScratchModel(Consumer<Criteria.Builder<Input, Output>> argument)
            throws IOException, ModelException, TranslateException {
        Path path = Paths.get("build/model");
        Files.createDirectories(path);
        Criteria.Builder<Input, Output> builder =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optEngine("RPC")
                        .optModelPath(path)
                        .optModelName("identity")
                        .optOption("hasParameter", "false")
                        .optBlock(Blocks.identityBlock());
        argument.accept(builder);
        try (ZooModel<Input, Output> model = builder.build().loadModel()) {
            return model.getTranslator().getClass().getSimpleName();
        }
    }
}
