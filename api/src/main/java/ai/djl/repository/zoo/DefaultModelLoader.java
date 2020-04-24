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
package ai.djl.repository.zoo;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDList;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.util.Progress;
import java.io.IOException;
import java.util.Map;

/** A {@link ModelLoader} loads a particular {@link ZooModel} from a local folder. */
public class DefaultModelLoader extends BaseModelLoader<NDList, NDList> {

    /**
     * Creates the model loader from the given repository.
     *
     * @param repository the repository to load the model from
     * @param mrl the mrl of the model to load
     */
    public DefaultModelLoader(Repository repository, MRL mrl) {
        super(repository, mrl, null, null);
    }

    /** {@inheritDoc} */
    @Override
    public Application getApplication() {
        return Application.UNDEFINED;
    }

    /** {@inheritDoc} */
    @Override
    public ZooModel<NDList, NDList> loadModel(
            Map<String, String> filters, Device device, Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optFilters(filters)
                        .optDevice(device)
                        .optProgress(progress)
                        .build();
        return loadModel(criteria);
    }
}
