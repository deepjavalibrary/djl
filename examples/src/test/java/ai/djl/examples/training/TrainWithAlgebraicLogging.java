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
package ai.djl.examples.training;

import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.examples.training.transferlearning.TrainResnetWithCifar10;
import ai.djl.testing.TestRequirements;
import ai.djl.training.TrainingResult;
import ai.djl.translate.TranslateException;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class TrainWithAlgebraicLogging {

    private static final int SEED = 1234;

    @Test
    public void testTrainMnist() throws ModelException, TranslateException, IOException {
        TestRequirements.engine("MXNet");

        Path pathActual = Paths.get("build/tmp/algebraiclog/TrainMnist.py");
        Files.createDirectories(pathActual.getParent());
        pathActual.toFile().delete();

        String[] args = new String[] {"-g", "1", "-m", "2", "-a", pathActual.toFile().toString()};

        TrainMnist.runExample(args);
        Path path = Paths.get("src/test/resources/algebraiclog/TrainMnist.py");

        try (InputStream is = Files.newInputStream(path);
                InputStream isActual = Files.newInputStream(pathActual)) {
            List<String> expected = Utils.readLines(is);
            List<String> actual = Utils.readLines(isActual);
            Assert.assertEquals(expected, actual);
        }
    }

    @Test
    public void testTrainResNetImperative() throws ModelException, IOException, TranslateException {
        TestRequirements.engine("MXNet");

        Path pathActual = Paths.get("build/tmp/algebraiclog/TrainResnetWithCifar10.py");
        Files.createDirectories(pathActual.getParent());
        pathActual.toFile().delete();

        // Limit max 4 gpu for cifar10 training to make it converge faster.
        // and only train 10 batch for unit test.
        String[] args = {
            "-e", "1", "-g", "4", "-m", "1", "-b", "111", "-a", pathActual.toFile().toString()
        };

        Engine.getInstance().setRandomSeed(SEED);
        TrainingResult result = TrainResnetWithCifar10.runExample(args);
        Assert.assertNotNull(result);

        Path path = Paths.get("src/test/resources/algebraiclog/TrainResnetWithCifar10.py");

        try (InputStream is = Files.newInputStream(path);
                InputStream isActual = Files.newInputStream(pathActual)) {
            List<String> expected = Utils.readLines(is);
            List<String> actual = Utils.readLines(isActual);
            Assert.assertEquals(expected, actual);
        }
    }
}
