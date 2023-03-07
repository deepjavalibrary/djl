/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.rx;

import ai.djl.ndarray.NDList;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import io.reactivex.rxjava3.core.Flowable;

import java.util.Iterator;

/**
 * A variation of a {@link SequentialBlock} that supports streaming with the {@link StreamingBlock}.
 */
public class StreamingSequentialBlock extends SequentialBlock implements StreamingBlock {

    /** Constructs a default {@link StreamingSequentialBlock}. */
    public StreamingSequentialBlock() {}

    /**
     * Constructs a copy as a {@link StreamingSequentialBlock} given a {@link SequentialBlock}
     * source.
     *
     * @param source the block to copy from
     */
    public StreamingSequentialBlock(SequentialBlock source) {
        super(source);
    }

    /** {@inheritDoc} */
    @Override
    public Flowable<NDList> forwardStream(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        return Flowable.fromIterable(() -> new StreamIterator(parameterStore, inputs, training));
    }

    private final class StreamIterator implements Iterator<NDList> {

        private int childIndex;
        private ParameterStore parameterStore;
        private NDList current;
        private boolean training;

        private StreamIterator(ParameterStore parameterStore, NDList inputs, boolean training) {
            this.parameterStore = parameterStore;
            this.current = inputs;
            this.training = training;
            childIndex = 0;
        }

        /** {@inheritDoc} */
        @Override
        public boolean hasNext() {
            return childIndex < children.size();
        }

        /** {@inheritDoc} */
        @Override
        public NDList next() {
            current =
                    children.get(childIndex++)
                            .getValue()
                            .forward(parameterStore, current, training);
            return current;
        }
    }
}
