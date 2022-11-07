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
package ai.djl.paddlepaddle.zoo.cv.imageclassification;

import ai.djl.Model;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.modality.cv.translator.ImageClassificationTranslatorFactory;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;

import java.util.Map;

/**
 * An {@link TranslatorFactory} that creates a {@link PpImageClassificationTranslatorFactory}
 * instance.
 */
public class PpImageClassificationTranslatorFactory extends ImageClassificationTranslatorFactory {

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public <I, O> Translator<I, O> newInstance(
            Class<I> input, Class<O> output, Model model, Map<String, ?> arguments) {
        ImageClassificationTranslator translator =
                ImageClassificationTranslator.builder()
                        .addTransform(new Resize(128, 128))
                        .addTransform(new ToTensor())
                        .addTransform(
                                new Normalize(
                                        new float[] {0.5f, 0.5f, 0.5f},
                                        new float[] {1.0f, 1.0f, 1.0f}))
                        .addTransform(nd -> nd.flip(0)) // RGB -> GBR
                        .build();
        return newInstance(input, output, translator);
    }
}
