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
package ai.djl.timeseries.translator;

import ai.djl.timeseries.ForeCast;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;

import java.time.LocalDateTime;
import java.util.Map;

/** Built-in {@code Translator} that provides default TimeSeriesTranslator config process. */
public abstract class BaseTimeSeriesTranslator implements Translator<TimeSeriesData, ForeCast> {

    protected int predictionLength;
    protected int contextLength;

    protected String freq;
    protected LocalDateTime startTime;

    private Batchifier batchifier;

    /**
     * Constructs a new {@code TimeSeriesTranslator} instance with the provided builder.
     *
     * @param builder the data to build with
     */
    protected BaseTimeSeriesTranslator(BaseBuilder<?> builder) {
        this.batchifier = builder.batchifier;
        this.freq = builder.freq;
        this.predictionLength = builder.predictionLength;
        // TODO: for inferring
        this.contextLength = builder.predictionLength;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return batchifier;
    }

    /**
     * A builder to extend for all classes extend the {@link BaseTimeSeriesTranslator}.
     *
     * @param <T> the concrete builder type
     */
    public abstract static class BaseBuilder<T extends BaseBuilder<T>> {
        protected Batchifier batchifier = Batchifier.STACK;
        protected int predictionLength;

        protected String freq;

        /**
         * Sets the {@link Batchifier} for the {@link Translator}.
         *
         * @param batchifier the {@link Batchifier} to be set
         * @return this builder
         */
        public T optBachifier(Batchifier batchifier) {
            this.batchifier = batchifier;
            return self();
        }

        protected abstract T self();

        protected void validate() {}

        protected void configPreProcess(Map<String, ?> arguments) {
            this.freq = ArgumentsUtil.stringValue(arguments, "freq", "D");
            this.predictionLength = ArgumentsUtil.intValue(arguments, "prediction_length");
            if (predictionLength <= 0) {
                throw new IllegalArgumentException(
                        "The value of `prediction_length` should be > 0");
            }
            if (arguments.containsKey("batchifier")) {
                batchifier = Batchifier.fromString((String) arguments.get("batchifier"));
            }
        }

        protected void configPostProcess(Map<String, ?> arguments) {}
    }
}
