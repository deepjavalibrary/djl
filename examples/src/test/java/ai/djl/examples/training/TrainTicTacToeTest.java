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
import org.apache.commons.cli.ParseException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class TrainTicTacToeTest {

    @Test
    public void testTrainTicTacToe() throws ParseException {
        if (Boolean.getBoolean("nightly")) {
            String[] args = new String[] {"-g", "1"};

            Engine.getInstance().setRandomSeed(1234);

            TrainingResult result = TrainTicTacToe.runExample(args);
            float winRate = result.getValidateEvaluation("winRate");
            Assert.assertTrue(winRate > 0.8f, "Win Rate: " + winRate);
        } else {
            String[] args = new String[] {"-g", "1", "-e", "1", "-m", "1"};

            TrainTicTacToe.runExample(args);
        }
    }
}
