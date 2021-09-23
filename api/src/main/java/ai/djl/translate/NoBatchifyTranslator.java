/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

/**
 * A {@link Translator} that does not use a {@link Batchifier}.
 *
 * <p>There are two major cases for avoiding the use of a {@link Batchifier}.
 *
 * <p>First, you want to translate between {@link ai.djl.training.dataset.Batch}es rather than
 * {@link ai.djl.training.dataset.Record}s. For example, you might go from String[] to Int[].
 *
 * <p>The second option is when using a model that does not use batching. Then, the model expects
 * only a single record at a time.
 *
 * @see Translator
 */
public interface NoBatchifyTranslator<I, O> extends Translator<I, O> {

    @Override
    /** {@inheritDoc} * */
    default Batchifier getBatchifier() {
        return null;
    }
}
