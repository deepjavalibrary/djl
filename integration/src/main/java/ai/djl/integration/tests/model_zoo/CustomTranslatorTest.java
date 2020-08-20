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
package ai.djl.integration.tests.model_zoo;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;
import ai.djl.util.ZipUtils;
import java.io.IOException;
import java.io.InputStream;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class CustomTranslatorTest {

    private Path modelDir = Paths.get("build/models/mlp");

    @BeforeClass
    public void setup() throws IOException, ModelNotFoundException, MalformedModelException {
        if (!"MXNet".equals(Engine.getInstance().getEngineName())) {
            return;
        }

        Utils.deleteQuietly(modelDir);
        Files.createDirectories(modelDir);
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optArtifactId("ai.djl.mxnet:mlp")
                        .build();

        try (ZooModel<Image, Classifications> model = ModelZoo.loadModel(criteria)) {
            Path symbolFile = modelDir.resolve("mlp-symbol.json");
            try (InputStream is = model.getArtifactAsStream("mlp-symbol.json")) {
                Files.copy(is, symbolFile, StandardCopyOption.REPLACE_EXISTING);
            }

            Path synsetFile = modelDir.resolve("synset.txt");
            try (InputStream is = model.getArtifactAsStream("synset.txt")) {
                Files.copy(is, synsetFile, StandardCopyOption.REPLACE_EXISTING);
            }

            Path paramFile = modelDir.resolve("mlp-0000.params");
            try (InputStream is = model.getArtifactAsStream("mlp-0000.params")) {
                Files.copy(is, paramFile, StandardCopyOption.REPLACE_EXISTING);
            }
        }
    }

    @AfterClass
    public void tearDown() {
        Utils.deleteQuietly(modelDir);
    }

    @Test
    public void testImageClassificationTranslator()
            throws IOException, ModelException, TranslateException {
        if (!"MXNet".equals(Engine.getInstance().getEngineName())) {
            return;
        }

        // load default translator with criteria
        Map<String, Object> arguments = new ConcurrentHashMap<>();
        arguments.put("width", "28");
        arguments.put("height", "28");
        arguments.put("flag", Image.Flag.GRAYSCALE.name());
        arguments.put("softmax", "true");
        runImageClassification(Application.CV.IMAGE_CLASSIFICATION, arguments);

        Path libDir = modelDir.resolve("lib");
        Path classesDir = libDir.resolve("classes");
        Files.createDirectories(classesDir);

        Properties prop = new Properties();
        prop.put("application", Application.CV.IMAGE_CLASSIFICATION.getPath());
        prop.put("width", "28");
        prop.put("height", "28");
        prop.put("flag", Image.Flag.GRAYSCALE.name());
        prop.put("softmax", "true");
        Path confFile = modelDir.resolve("serving.properties");
        try (Writer writer = Files.newBufferedWriter(confFile)) {
            prop.store(writer, "");
        }
        //
        runImageClassification(Application.UNDEFINED, null);

        Path libsDir = modelDir.resolve("libs");
        Files.move(libDir, libsDir, StandardCopyOption.REPLACE_EXISTING);
        classesDir = libsDir.resolve("classes");
        Path srcFile = Paths.get("src/test/translator/MyTranslator.java");
        Path destFile = classesDir.resolve("MyTranslator.java");
        Files.copy(srcFile, destFile, StandardCopyOption.REPLACE_EXISTING);

        // load translator from classes folder
        runImageClassification(Application.UNDEFINED, null);

        Path jarFile = libsDir.resolve("example.jar");
        ZipUtils.zip(classesDir, jarFile, false);
        Utils.deleteQuietly(classesDir);

        // load translator from jar file
        runImageClassification(Application.UNDEFINED, null);
    }

    @Test
    public void testSsdTranslator() throws IOException, ModelException, TranslateException {
        if (!"MXNet".equals(Engine.getInstance().getEngineName())) {
            return;
        }

        Criteria<Image, DetectedObjects> c =
                Criteria.builder()
                        .setTypes(Image.class, DetectedObjects.class)
                        .optArtifactId("ai.djl.mxnet:ssd")
                        .build();
        String modelUrl;
        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(c)) {
            modelUrl = model.getModelPath().toUri().toURL().toString();
        }

        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .optModelUrls(modelUrl)
                        .optArgument("synset", "classes.txt")
                        .optModelName("ssd_512_resnet50_v1_voc")
                        .build();

        Path imageFile = Paths.get("../examples/src/test/resources/dog_bike_car.jpg");
        byte[] data;
        try (InputStream is = Files.newInputStream(imageFile)) {
            data = Utils.toByteArray(is);
        }

        try (ZooModel<Input, Output> model = ModelZoo.loadModel(criteria);
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input("1");
            input.addData("data", data);
            Output output = predictor.predict(input);
            Assert.assertEquals(output.getRequestId(), "1");
            String content = new String(output.getContent(), StandardCharsets.UTF_8);
            Classifications result = JsonUtils.GSON.fromJson(content, Classifications.class);
            Assert.assertEquals(result.best().getClassName(), "car");
        }
    }

    private void runImageClassification(Application application, Map<String, Object> arguments)
            throws IOException, ModelException, TranslateException {
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optApplication(application)
                        .optArguments(arguments)
                        .optModelUrls(modelDir.toString())
                        .build();

        Path imageFile = Paths.get("../examples/src/test/resources/0.png");
        byte[] data;
        try (InputStream is = Files.newInputStream(imageFile)) {
            data = Utils.toByteArray(is);
        }

        try (ZooModel<Input, Output> model = ModelZoo.loadModel(criteria);
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input("1");
            input.addData("data", data);
            Output output = predictor.predict(input);
            Assert.assertEquals(output.getRequestId(), "1");
            String content = new String(output.getContent(), StandardCharsets.UTF_8);
            Classifications result = JsonUtils.GSON.fromJson(content, Classifications.class);
            Assert.assertEquals(result.best().getClassName(), "0");
        }
    }
}
