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

import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.examples.training.util.Arguments;
import ai.djl.modality.rl.agent.EpsilonGreedy;
import ai.djl.modality.rl.agent.QAgent;
import ai.djl.modality.rl.agent.RlAgent;
import ai.djl.modality.rl.env.RlEnv.Step;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.LinearTracker;
import ai.djl.training.tracker.Tracker;
import java.io.IOException;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An example of training reinforcement learning using {@link TicTacToe} and a {@link QAgent}.
 *
 * <p>Note that the current setup, for simplicity, has one agent playing both sides to ensure X
 * always wins.
 */
public final class TrainTicTacToe {

    private static final Logger logger = LoggerFactory.getLogger(TrainTicTacToe.class);

    private TrainTicTacToe() {}

    public static void main(String[] args) throws IOException {
        TrainTicTacToe.runExample(args);
    }

    public static TrainingResult runExample(String[] args) throws IOException {
        Arguments arguments = new Arguments().parseArgs(args);
        if (arguments == null) {
            return null;
        }

        int epoch = arguments.getEpoch();
        int batchSize = arguments.getBatchSize();
        int replayBufferSize = 1024;
        int gamesPerEpoch = 128;
        // Validation is deterministic, thus one game is enough
        int validationGamesPerEpoch = 1;
        float rewardDiscount = 0.9f;

        if (arguments.getLimit() != Long.MAX_VALUE) {
            gamesPerEpoch = Math.toIntExact(arguments.getLimit());
        }

        TicTacToe game = new TicTacToe(NDManager.newBaseManager(), batchSize, replayBufferSize);

        Block block = getBlock();

        try (Model model = Model.newInstance("tic-tac-toe")) {
            model.setBlock(block);

            DefaultTrainingConfig config = setupTrainingConfig();
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(
                        new Shape(batchSize, 9), new Shape(batchSize), new Shape(batchSize));

                trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));

                // Constructs the agent to train and play with
                RlAgent agent = new QAgent(trainer, rewardDiscount);
                Tracker exploreRate =
                        LinearTracker.builder()
                                .setBaseValue(0.9f)
                                .optSlope(-.9f / (epoch * gamesPerEpoch * 7))
                                .optMinValue(0.01f)
                                .build();
                agent = new EpsilonGreedy(agent, exploreRate);

                float validationWinRate = 0;
                float trainWinRate = 0;
                for (int i = 0; i < epoch; i++) {
                    int trainingWins = 0;
                    for (int j = 0; j < gamesPerEpoch; j++) {
                        float result = game.runEnvironment(agent, true);
                        Step[] batchSteps = game.getBatch();
                        agent.trainBatch(batchSteps);
                        trainer.step();

                        // Record if the game was won
                        if (result > 0) {
                            trainingWins++;
                        }
                    }

                    trainWinRate = (float) trainingWins / gamesPerEpoch;
                    logger.info("Training wins: {}", trainWinRate);

                    trainer.notifyListeners(listener -> listener.onEpoch(trainer));

                    // Counts win rate after playing {validationGamesPerEpoch} games
                    int validationWins = 0;
                    for (int j = 0; j < validationGamesPerEpoch; j++) {
                        float result = game.runEnvironment(agent, false);
                        if (result > 0) {
                            validationWins++;
                        }
                    }

                    validationWinRate = (float) validationWins / validationGamesPerEpoch;
                    logger.info("Validation wins: {}", validationWinRate);
                }

                trainer.notifyListeners(listener -> listener.onTrainingEnd(trainer));

                TrainingResult trainingResult = trainer.getTrainingResult();
                trainingResult.getEvaluations().put("validate_winRate", validationWinRate);
                trainingResult.getEvaluations().put("train_winRate", trainWinRate);

                model.save(Paths.get("build/model"), "tictactoe");
                return trainingResult;
            }
        }
    }

    public static Block getBlock() {
        return new SequentialBlock()
                .add(
                        arrays -> {
                            NDArray board = arrays.get(0); // Shape(N, 9)
                            NDArray turn = arrays.get(1).reshape(-1, 1); // Shape(N, 1)
                            NDArray action = arrays.get(2).reshape(-1, 1); // Shape(N, 1)

                            // Concatenate to a combined vector of Shape(N, 11)
                            NDArray combined = NDArrays.concat(new NDList(board, turn, action), 1);

                            return new NDList(combined.toType(DataType.FLOAT32, true));
                        })
                .add(new Mlp(11, 1, new int[] {20, 10}));
    }

    public static DefaultTrainingConfig setupTrainingConfig() {
        return new DefaultTrainingConfig(Loss.l2Loss())
                .addTrainingListeners(TrainingListener.Defaults.basic())
                .optOptimizer(
                        Adam.builder().optLearningRateTracker(Tracker.fixed(0.0001F)).build());
    }
}
