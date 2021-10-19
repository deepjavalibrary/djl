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
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.wrapper.FileTranslator;
import ai.djl.modality.cv.translator.wrapper.InputStreamTranslator;
import ai.djl.modality.cv.translator.wrapper.UrlTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.util.Map;

/** An {@link TranslatorFactory} that creates a {@link YoloV5Translator} instance. */
public class YoloV5TranslatorFactory extends ObjectDetectionTranslatorFactory {

    /** {@inheritDoc} */
    @Override
    public Translator<?, ?> newInstance(
            Class<?> input, Class<?> output, Model model, Map<String, ?> arguments)
            throws TranslateException {
        if (input == Image.class && output == DetectedObjects.class) {
            return YoloV5Translator.builder(arguments).build();
        } else if (input == Path.class && output == DetectedObjects.class) {
            return new FileTranslator<>(YoloV5Translator.builder(arguments).build());
        } else if (input == URL.class && output == DetectedObjects.class) {
            return new UrlTranslator<>(YoloV5Translator.builder(arguments).build());
        } else if (input == InputStream.class && output == DetectedObjects.class) {
            return new InputStreamTranslator<>(YoloV5Translator.builder(arguments).build());
        } else if (input == Input.class && output == Output.class) {
            return new ImageServingTranslator(YoloV5Translator.builder(arguments).build());
        }
        throw new IllegalArgumentException("Unsupported input/output types.");
    }
}
