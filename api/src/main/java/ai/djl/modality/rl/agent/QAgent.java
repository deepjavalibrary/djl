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
package ai.djl.modality.rl.agent;

import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.modality.rl.env.RlEnv.Step;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.listener.TrainingListener.BatchData;
import ai.djl.translate.Batchifier;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;

/**
 * An {@link RlAgent} that implements Q or Deep-Q Learning.
 *
 * <p>Deep-Q Learning estimates the total reward that will be given until the environment ends in a
 * particular state after taking a particular action. Then, it is trained by ensuring that the
 * prediction before taking the action match what would be predicted after taking the action. More
 * information can be found in the <a
 * href="https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf">paper</a>.
 *
 * <p>It is one of the earliest successful techniques for reinforcement learning with Deep learning.
 * It is also a good introduction to the field. However, many better techniques are commonly used
 * now.
 */
public class QAgent implements RlAgent {

    private Trainer trainer;
    private float rewardDiscount;
    private Batchifier batchifier;

    /**
     * Constructs a {@link QAgent}.
     *
     * <p>It uses the {@link ai.djl.translate.StackBatchifier} as the default batchifier.
     *
     * @param trainer the trainer for the model to learn
     * @param rewardDiscount the reward discount to apply to rewards from future states
     */
    public QAgent(Trainer trainer, float rewardDiscount) {
        this(trainer, rewardDiscount, Batchifier.STACK);
    }

    /**
     * Constructs a {@link QAgent} with a custom {@link Batchifier}.
     *
     * @param trainer the trainer for the model to learn
     * @param rewardDiscount the reward discount to apply to rewards from future states
     * @param batchifier the batchifier to join inputs with
     */
    public QAgent(Trainer trainer, float rewardDiscount, Batchifier batchifier) {
        this.trainer = trainer;
        this.rewardDiscount = rewardDiscount;
        this.batchifier = batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public NDList chooseAction(RlEnv env, boolean training) {
        ActionSpace actionSpace = env.getActionSpace();
        NDList[] inputs = buildInputs(env.getObservation(), actionSpace);
        NDArray actionScores =
                trainer.evaluate(batchifier.batchify(inputs)).singletonOrThrow().squeeze(-1);
        int bestAction = Math.toIntExact(actionScores.argMax().getLong());
        return actionSpace.get(bestAction);
    }

    /** {@inheritDoc} */
    @Override
    public void trainBatch(Step[] batchSteps) {
        BatchData batchData =
                new BatchData(null, new ConcurrentHashMap<>(), new ConcurrentHashMap<>());
        for (Step step : batchSteps) {

            NDList[] preInput =
                    buildInputs(
                            step.getPreObservation(), Collections.singletonList(step.getAction()));
            NDList[] postInputs = buildInputs(step.getPostObservation(), step.getPostActionSpace());
            NDList[] allInputs =
                    Stream.concat(Arrays.stream(preInput), Arrays.stream(postInputs))
                            .toArray(NDList[]::new);

            try (GradientCollector collector = trainer.newGradientCollector()) {
                NDArray results =
                        trainer.forward(batchifier.batchify(allInputs))
                                .singletonOrThrow()
                                .squeeze(-1);
                NDList preQ = new NDList(results.get(0));
                NDList postQ;
                if (step.isDone()) {
                    postQ = new NDList(step.getReward());
                } else {
                    NDArray bestAction = results.get("1:").max();
                    postQ = new NDList(bestAction.mul(rewardDiscount).add(step.getReward()));
                }
                NDArray lossValue = trainer.getLoss().evaluate(postQ, preQ);
                collector.backward(lossValue);
                batchData.getLabels().put(postQ.get(0).getDevice(), postQ);
                batchData.getPredictions().put(preQ.get(0).getDevice(), preQ);
            }
        }

        trainer.notifyListeners(listener -> listener.onTrainingBatch(trainer, batchData));
    }

    private NDList[] buildInputs(NDList observation, List<NDList> actions) {
        NDList[] inputs = new NDList[actions.size()];
        for (int i = 0; i < actions.size(); i++) {
            NDList nextData = new NDList().addAll(observation).addAll(actions.get(i));
            inputs[i] = nextData;
        }
        return inputs;
    }
}
