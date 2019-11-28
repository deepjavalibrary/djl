/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.repository.zoo;

import java.util.List;
import java.util.ServiceLoader;

/** An interface represents a collection of models. */
public interface ModelZoo {

    /**
     * Returns the {@code ModelZoo} with the given name.
     *
     * @param name the name of ModelZoo to retrieve
     * @return the instance of {@code ModelZoo}
     * @throws ZooProviderNotFoundException when the provider cannot be found
     * @see ZooProvider
     */
    static ModelZoo getModelZoo(String name) {
        ServiceLoader<ZooProvider> providers = ServiceLoader.load(ZooProvider.class);
        for (ZooProvider provider : providers) {
            if (provider.getName().equals(name)) {
                return provider.getModelZoo();
            }
        }
        throw new ZooProviderNotFoundException("ZooProvider not found: " + name);
    }

    /**
     * Lists the available model families in the ModelZoo.
     *
     * @return the list of all available model families
     */
    List<ModelLoader<?, ?>> getModelLoaders();

    /**
     * Gets the {@link ModelLoader} based on the model name.
     *
     * @param name the name of the model
     * @param <I> the input data type for preprocessing
     * @param <O> the output data type after postprocessing
     * @return the {@link ModelLoader} of the model
     * @throws ModelNotFoundException when the model cannot be found
     */
    <I, O> ModelLoader<I, O> getModelLoader(String name) throws ModelNotFoundException;
}
