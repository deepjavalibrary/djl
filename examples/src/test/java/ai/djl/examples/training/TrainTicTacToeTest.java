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
package ai.djl.examples.training;

import ai.djl.engine.Engine;
import ai.djl.training.TrainingResult;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class TrainTicTacToeTest {

    @Test
    public void testTrainTicTacToe() throws IOException {
        if (Boolean.getBoolean("nightly")) {
            String[] args = new String[] {"-g", "1", "-e", "6"};
            Engine.getInstance().setRandomSeed(1234);

            TrainingResult result = TrainTicTacToe.runExample(args);
            Assert.assertNotNull(result);

            float trainWinRate = result.getTrainEvaluation("winRate");
            Assert.assertTrue(trainWinRate > 0.8f, "Train win Rate: " + trainWinRate);

            float validationWinRate = result.getValidateEvaluation("winRate");
            // TicTacToe game run is deterministic when training == false, thus winRate == 0 | 1
            Assert.assertEquals(validationWinRate, 1f, "Validation Win Rate: " + validationWinRate);
        } else {
            String[] args = new String[] {"-g", "1", "-e", "1", "-m", "1"};
            TrainingResult result = TrainTicTacToe.runExample(args);
            Assert.assertNotNull(result);
        }
    }
}
