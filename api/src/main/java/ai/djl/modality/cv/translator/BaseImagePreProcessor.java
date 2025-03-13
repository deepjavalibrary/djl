/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv.translator;

import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDList;
import ai.djl.translate.TranslatorContext;

/** A {@code BaseImageTranslator} that only handles pre-processing inputs. */
public class BaseImagePreProcessor extends BaseImageTranslator<Void> {

    /**
     * Constructs an {@code ImageTranslator} with the provided builder.
     *
     * @param builder the data to build with
     */
    public BaseImagePreProcessor(BaseBuilder<?> builder) {
        super(builder);
    }

    /** {@inheritDoc} */
    @Override
    public Void processOutput(TranslatorContext ctx, NDList list) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        return super.processInput(ctx, input);
    }
}
