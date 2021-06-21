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
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.translator.ImageClassificationTranslatorFactory;
import ai.djl.modality.cv.translator.wrapper.FileTranslator;
import ai.djl.modality.cv.translator.wrapper.InputStreamTranslator;
import ai.djl.modality.cv.translator.wrapper.UrlTranslator;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.util.Map;

/** An {@link TranslatorFactory} that creates a {@link PpWordRotateTranslatorFactory} instance. */
public class PpWordRotateTranslatorFactory extends ImageClassificationTranslatorFactory {

    /** {@inheritDoc} */
    @Override
    public Translator<?, ?> newInstance(
            Class<?> input, Class<?> output, Model model, Map<String, ?> arguments) {
        if (input == Image.class && output == Classifications.class) {
            return new PpWordRotateTranslator();
        } else if (input == Path.class && output == Classifications.class) {
            return new FileTranslator<>(new PpWordRotateTranslator());
        } else if (input == URL.class && output == Classifications.class) {
            return new UrlTranslator<>(new PpWordRotateTranslator());
        } else if (input == InputStream.class && output == Classifications.class) {
            return new InputStreamTranslator<>(new PpWordRotateTranslator());
        }
        throw new IllegalArgumentException("Unsupported input/output types.");
    }
}
