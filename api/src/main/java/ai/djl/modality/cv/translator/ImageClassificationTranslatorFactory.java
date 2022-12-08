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
package ai.djl.modality.cv.translator;

import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;

import java.util.Map;

/** A {@link TranslatorFactory} that creates an {@link ImageClassificationTranslator}. */
public class ImageClassificationTranslatorFactory
        extends BaseImageTranslatorFactory<Classifications> {

    /** {@inheritDoc} */
    @Override
    protected Translator<Image, Classifications> buildBaseTranslator(
            Model model, Map<String, ?> arguments) {
        return ImageClassificationTranslator.builder(arguments).build();
    }

    /** {@inheritDoc} */
    @Override
    public Class<Classifications> getBaseOutputType() {
        return Classifications.class;
    }
}
