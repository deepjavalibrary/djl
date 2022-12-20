/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.inference.stablediffusion;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;

public class PndmScheduler {
    private static final int TRAIN_TIMESTEPS = 1000;
    private static final float BETA_START = 0.00085f;
    private static final float BETA_END = 0.012f;
    private NDManager manager;
    private NDArray alphasCumProd;
    private float finalAlphaCumProd;
    private int counter;
    private NDArray curSample;
    private NDList ets;
    private int stepSize;
    private int[] timesteps;

    public PndmScheduler(NDManager mgr) {
        manager = mgr;
        NDArray betas =
                manager.linspace(
                                (float) Math.sqrt(BETA_START),
                                (float) Math.sqrt(BETA_END),
                                TRAIN_TIMESTEPS)
                        .square();
        NDArray alphas = manager.ones(betas.getShape()).sub(betas);
        alphasCumProd = alphas.cumProd(0);
        finalAlphaCumProd = alphasCumProd.get(0).toFloatArray()[0];
        ets = new NDList();
    }

    public NDArray addNoise(NDArray latent, NDArray noise, int timesteps) {
        float alphaProd = alphasCumProd.get(timesteps).toFloatArray()[0];
        float sqrtOneMinusAlphaProd = (float) Math.sqrt(1 - alphaProd);
        latent = latent.mul(alphaProd).add(noise.mul(sqrtOneMinusAlphaProd));
        return latent;
    }

    public void initTimesteps(int inferenceSteps, int offset) {
        stepSize = TRAIN_TIMESTEPS / inferenceSteps;
        NDArray timestepsNd = manager.arange(0, inferenceSteps).mul(stepSize).add(offset);

        // np.concatenate([self._timesteps[:-1], self._timesteps[-2:-1],
        // self._timesteps[-1:]])[::-1]
        NDArray part1 = timestepsNd.get(new NDIndex(":-1"));
        NDArray part2 = timestepsNd.get(new NDIndex("-2:-1"));
        NDArray part3 = timestepsNd.get(new NDIndex("-1:"));
        NDList list = new NDList();
        list.add(part1);
        list.add(part2);
        list.add(part3);
        // timesteps = timesteps.get(new NDIndex("::-1"));
        timesteps = NDArrays.concat(list).flatten().flip(0).toIntArray();
    }

    public NDArray step(NDArray modelOutput, int timestep, NDArray sample) {
        int prevTimestep = timestep - stepSize;
        if (counter != 1) {
            ets.add(modelOutput);
        } else {
            prevTimestep = timestep;
            timestep -= stepSize;
        }

        if (ets.size() == 1 && counter == 0) {
            curSample = sample;
        } else if (ets.size() == 1 && counter == 1) {
            modelOutput = modelOutput.add(ets.get(0)).div(2);
            sample = curSample;
        } else if (ets.size() == 2) {
            NDArray firstModel = ets.get(ets.size() - 1).mul(3);
            NDArray secondModel = ets.get(ets.size() - 2).mul(-1);
            modelOutput = firstModel.add(secondModel);
            modelOutput = modelOutput.div(2);
        } else if (ets.size() == 3) {
            NDArray firstModel = ets.get(ets.size() - 1).mul(23);
            NDArray secondModel = ets.get(ets.size() - 2).mul(-16);
            NDArray thirdModel = ets.get(ets.size() - 3).mul(5);
            modelOutput = firstModel.add(secondModel).add(thirdModel);
            modelOutput = modelOutput.div(12);
        } else {
            NDArray firstModel = ets.get(ets.size() - 1).mul(55);
            NDArray secondModel = ets.get(ets.size() - 2).mul(-59);
            NDArray thirdModel = ets.get(ets.size() - 3).mul(37);
            NDArray fourthModel = ets.get(ets.size() - 4).mul(-9);
            modelOutput = firstModel.add(secondModel).add(thirdModel).add(fourthModel);
            modelOutput = modelOutput.div(24);
        }

        NDArray prevSample = getPrevSample(sample, timestep, prevTimestep, modelOutput);
        prevSample.setName("prev_sample");
        counter++;

        return prevSample;
    }

    public int[] getTimesteps() {
        return timesteps;
    }

    public void setTimesteps(int[] timesteps) {
        this.timesteps = timesteps;
    }

    private NDArray getPrevSample(
            NDArray sample, int timestep, int prevTimestep, NDArray modelOutput) {
        float alphaProdT = alphasCumProd.toFloatArray()[timestep];
        float alphaProdTPrev;

        if (prevTimestep >= 0) {
            alphaProdTPrev = alphasCumProd.toFloatArray()[prevTimestep];
        } else {
            alphaProdTPrev = finalAlphaCumProd;
        }

        float betaProdT = 1 - alphaProdT;
        float betaProdTPrev = 1 - alphaProdTPrev;

        float sampleCoeff = (float) Math.sqrt(alphaProdTPrev / alphaProdT);
        float modelOutputCoeff =
                alphaProdT * (float) Math.sqrt(betaProdTPrev)
                        + (float) Math.sqrt(alphaProdT * betaProdT * alphaProdTPrev);

        sample = sample.mul(sampleCoeff);
        modelOutput = modelOutput.mul(alphaProdTPrev - alphaProdT).div(modelOutputCoeff).neg();
        return sample.add(modelOutput);
    }
}
