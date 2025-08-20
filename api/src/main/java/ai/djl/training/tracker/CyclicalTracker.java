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
package ai.djl.training.tracker;

import ai.djl.util.Preconditions;

/**
 * {@code CyclicalTracker} is an implementation of {@link Tracker} which is a policy of learning
 * rate adjustment that increases the learning rate off a base value in a cyclical nature, as
 * detailed in the paper <a href="https://arxiv.org/abs/1506.01186">Cyclical Learning Rates for
 * Training Neural Networks</a>.
 */
public class CyclicalTracker implements Tracker {

    private float baseValue;
    private float maxValue;
    private int stepSizeUp;
    private int stepSizeDown;
    private int totalSize;
    private float stepRatio;
    private ScaleFunction scaleFunction;
    private boolean scaleModeCycle;

    /**
     * Creates a new instance of {@code CyclicalTracker}.
     *
     * @param builder the builder to create a new instance of {@code CyclicalTracker}
     */
    public CyclicalTracker(Builder builder) {
        this.baseValue = builder.baseValue;
        this.maxValue = builder.maxValue;
        this.stepSizeUp = builder.stepSizeUp;
        this.stepSizeDown = builder.stepSizeDown > 0 ? builder.stepSizeDown : builder.stepSizeUp;
        this.totalSize = this.stepSizeUp + this.stepSizeDown;
        this.stepRatio = (float) this.stepSizeUp / this.totalSize;
        if (builder.scaleFunction != null) {
            this.scaleFunction = builder.scaleFunction;
            this.scaleModeCycle = builder.scaleModeCycle;
        } else {
            switch (builder.mode) {
                case TRIANGULAR:
                    this.scaleFunction = new TriangularScaleFunction();
                    this.scaleModeCycle = true;
                    break;
                case TRIANGULAR2:
                    this.scaleFunction = new Triangular2ScaleFunction();
                    this.scaleModeCycle = true;
                    break;
                case EXP_RANGE:
                    this.scaleFunction = new ExpRangeScaleFunction(builder.gamma);
                    this.scaleModeCycle = false;
                    break;
                default:
                    throw new UnsupportedOperationException("Unsupported Cyclical mode.");
            }
        }
    }

    /**
     * Creates a new builder.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** {@inheritDoc} */
    @Override
    public float getNewValue(int numUpdate) {
        int cycle = (int) Math.floor(1 + (float) numUpdate / this.totalSize);
        float x = 1 + (float) numUpdate / this.totalSize - cycle;
        float scaleFactor;
        float res;
        if (x < this.stepRatio) {
            scaleFactor = x / this.stepRatio;
        } else {
            scaleFactor = (x - 1) / (this.stepRatio - 1);
        }
        float baseHeight = (this.maxValue - this.baseValue) * scaleFactor;
        if (this.scaleModeCycle) {
            res = this.baseValue + baseHeight * this.scaleFunction.func(cycle);
        } else {
            res = this.baseValue + baseHeight * this.scaleFunction.func(numUpdate);
        }
        return res;
    }

    /** The Builder to construct an {@link CyclicalTracker} object. */
    public static final class Builder {

        private float baseValue = 0.001f;
        private float maxValue = 0.006f;
        private int stepSizeUp = 2000;
        private int stepSizeDown;
        private CyclicalMode mode = CyclicalMode.TRIANGULAR;
        private ScaleFunction scaleFunction;
        private boolean scaleModeCycle = true;
        private float gamma = 1f;

        private Builder() {}

        /**
         * Sets the initial value. Default: 0.001.
         *
         * @param baseValue the initial value
         * @return this {@code Builder}
         */
        public Builder optBaseValue(float baseValue) {
            this.baseValue = baseValue;
            return this;
        }

        /**
         * Sets the initial value. Default: 0.006.
         *
         * @param maxValue the max value
         * @return this {@code Builder}
         */
        public Builder optMaxValue(float maxValue) {
            this.maxValue = maxValue;
            return this;
        }

        /**
         * Sets the number of iterations in up half of cycle. Default: 2000.
         *
         * @param stepSizeUp number of iterations in up half of cycle
         * @return this {@code Builder}
         */
        public Builder optStepSizeUp(int stepSizeUp) {
            this.stepSizeUp = stepSizeUp;
            return this;
        }

        /**
         * Sets the number of iterations in up half of cycle. If {@code stepSizeDown} equals 0, it
         * is set to {@code stepSizeUp}. Default: 0.
         *
         * @param stepSizeDown number of iterations in the down half of cycle
         * @return this {@code Builder}
         */
        public Builder optStepSizeDown(int stepSizeDown) {
            this.stepSizeDown = stepSizeDown;
            return this;
        }

        /**
         * Sets the cyclical mode. Can be {@code CyclicalMode.TRIANGULAR}, {@code
         * CyclicalMode.TRIANGULAR2} or {@code CyclicalMode.EXP_RANGE}. Values correspond to
         * policies detailed above.
         *
         * @param mode cyclical mode
         * @return this {@code Builder}
         */
        public Builder optMode(CyclicalMode mode) {
            this.mode = mode;
            return this;
        }

        /**
         * Constant in {@code CyclicalMode.EXP_RANGE} scaling function. Default: 1.
         *
         * @param gamma constant in 'exp_range' scaling function: gamma^cycleIterations
         * @return this {@code Builder}
         */
        public Builder optGamma(float gamma) {
            this.gamma = gamma;
            return this;
        }

        /**
         * Custom scaling function. If set, {@code mode} will be ignored. Default: null.
         *
         * @param scaleFunction Custom scaling function
         * @return this {@code Builder}
         */
        public Builder optScaleFunction(ScaleFunction scaleFunction) {
            this.scaleFunction = scaleFunction;
            return this;
        }

        /**
         * Defines whether scaling function is evaluated on cycle number or update number. Default:
         * true.
         *
         * @param scaleModeCycle if true then scaling function is evaluated on cycle number, else on
         *     update number
         * @return this {@code Builder}
         */
        public Builder optScaleModeCycle(boolean scaleModeCycle) {
            this.scaleModeCycle = scaleModeCycle;
            return this;
        }

        /**
         * Builds a {@link CyclicalTracker} block.
         *
         * @return the {@link CyclicalTracker} block
         */
        public CyclicalTracker build() {
            Preconditions.checkArgument(baseValue > 0, "baseValue has to be positive!");
            Preconditions.checkArgument(maxValue > 0, "maxValue has to be positive!");
            Preconditions.checkArgument(
                    baseValue <= maxValue, "baseValue has to lower than maxValue!");
            Preconditions.checkArgument(stepSizeUp >= 1, "stepSizeUp has to be positive!");
            Preconditions.checkArgument(stepSizeDown >= 0, "stepSizeUp cannot be negative!");
            Preconditions.checkArgument(
                    gamma >= 0f && gamma <= 1f, "gamma has to be between 0 and 1!");
            return new CyclicalTracker(this);
        }
    }

    /**
     * {@code CyclicalTracker} provides three predefined cyclical modes and can be selected by this
     * enum.
     *
     * <ul>
     *   <li>TRIANGULAR: A basic triangular cycle without amplitude scaling.
     *   <li>TRIANGULAR2: A basic triangular cycle that scales initial amplitude by half each cycle.
     *   <li>EXP_RANGE: A cycle that scales initial amplitude by (gamma<sup>cycleIterations</sup>)
     *       at each cycle iteration.
     * </ul>
     */
    public enum CyclicalMode {
        TRIANGULAR,
        TRIANGULAR2,
        EXP_RANGE
    }

    /** {@code ScaleFunction} is an interface to implement a custom scale function. */
    public interface ScaleFunction {

        /**
         * Custom scaling policy. Return value has to satisfy 0&lt;=func(steps)&lt;=1 for all
         * steps&gt;1.
         *
         * @param steps current cycles if {@code scaleModeCycle} is true; input update number if it
         *     is false
         * @return scale ratio
         */
        float func(int steps);
    }

    private static final class TriangularScaleFunction implements ScaleFunction {

        /** {@inheritDoc} */
        @Override
        public float func(int steps) {
            return 1f;
        }
    }

    private static final class Triangular2ScaleFunction implements ScaleFunction {

        /** {@inheritDoc} */
        @Override
        public float func(int steps) {
            return (float) (1 / (Math.pow(2f, steps - 1)));
        }
    }

    private static final class ExpRangeScaleFunction implements ScaleFunction {

        float gamma;

        ExpRangeScaleFunction(float gamma) {
            this.gamma = gamma;
        }

        /** {@inheritDoc} */
        @Override
        public float func(int steps) {
            return (float) Math.pow(this.gamma, steps);
        }
    }
}
