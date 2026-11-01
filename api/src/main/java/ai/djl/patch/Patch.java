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
package ai.djl.patch;

import ai.djl.Model;

/**
 * A method for modifying a {@link Model}.
 *
 * <p>The most standard form is the {@link ParamPatch}.
 */
public abstract class Patch implements AutoCloseable {

    /**
     * Applies this patch to a model.
     *
     * @param model the model to update with the patch
     */
    public abstract void apply(Model model);

    /** {@inheritDoc} */
    @Override
    public abstract void close();
}
