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
package ai.djl.translate;

import java.util.List;

/**
 * Represents a class that can be ensembled (or averaged).
 *
 * <p>Typically, ensembling is used for the output of models/translators. By averaging multiple
 * models, it is often possible to get greater accuracy then running each model individually.
 */
public interface Ensembleable {

    /**
     * Finds the ensemble of a list of outputs.
     *
     * @param outputs the outputs to ensemble. It uses the caller class to determine how to
     *     ensemble, but should include the caller object in the list.
     * @return the ensembled (averaged) output
     * @param <T> the type of object to ensemble. Usually also the type returned
     */
    <T extends Ensembleable> Ensembleable ensemble(List<T> outputs);
}
