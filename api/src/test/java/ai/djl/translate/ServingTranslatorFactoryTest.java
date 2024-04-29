/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.repository.zoo.ZooModel;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ServingTranslatorFactoryTest {

    @Test
    public void test() throws IOException, TranslateException, ModelException {
        Path path = Paths.get("build/model");
        Files.createDirectories(path);
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(path)
                        .optModelName("identity")
                        .optArgument("application", "image_classification")
                        .optOption("hasParameter", "false")
                        .optBlock(Blocks.identityBlock())
                        .build();

        try (ZooModel<Input, Output> model = criteria.loadModel()) {
            String className = model.getTranslator().getClass().getSimpleName();
            Assert.assertEquals(className, "ImageServingTranslator");
        }

        criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(path)
                        .optModelName("identity")
                        // HF style task name
                        .optArgument("task", "fill-mask")
                        .optOption("hasParameter", "false")
                        .optBlock(Blocks.identityBlock())
                        .build();

        try (ZooModel<Input, Output> model = criteria.loadModel()) {
            String className = model.getTranslator().getClass().getSimpleName();
            // Tokenizer is not in class path fallback to NoopServingTranslator
            Assert.assertEquals(className, "NoopServingTranslator");
        }

        criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(path)
                        .optModelName("identity")
                        // HF style task name
                        .optArgument("task", "question-answering")
                        .optOption("hasParameter", "false")
                        .optBlock(Blocks.identityBlock())
                        .build();

        try (ZooModel<Input, Output> model = criteria.loadModel()) {
            String className = model.getTranslator().getClass().getSimpleName();
            // Tokenizer is not in class path fallback to NoopServingTranslator
            Assert.assertEquals(className, "NoopServingTranslator");
        }

        criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(path)
                        .optModelName("identity")
                        // HF style task name
                        .optArgument("task", "sentence-similarity")
                        .optOption("hasParameter", "false")
                        .optBlock(Blocks.identityBlock())
                        .build();

        try (ZooModel<Input, Output> model = criteria.loadModel()) {
            String className = model.getTranslator().getClass().getSimpleName();
            // Tokenizer is not in class path fallback to NoopServingTranslator
            Assert.assertEquals(className, "NoopServingTranslator");
        }

        criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(path)
                        .optModelName("identity")
                        // HF style task name
                        .optArgument("task", "text-classification")
                        .optOption("hasParameter", "false")
                        .optBlock(Blocks.identityBlock())
                        .build();

        try (ZooModel<Input, Output> model = criteria.loadModel()) {
            String className = model.getTranslator().getClass().getSimpleName();
            // Tokenizer is not in class path fallback to NoopServingTranslator
            Assert.assertEquals(className, "NoopServingTranslator");
        }

        criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(path)
                        .optModelName("identity")
                        // HF style task name
                        .optArgument("task", "token-classification")
                        .optOption("hasParameter", "false")
                        .optBlock(Blocks.identityBlock())
                        .build();

        try (ZooModel<Input, Output> model = criteria.loadModel()) {
            String className = model.getTranslator().getClass().getSimpleName();
            // Tokenizer is not in class path fallback to NoopServingTranslator
            Assert.assertEquals(className, "NoopServingTranslator");
        }
    }
}
