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
package ai.djl.translate;

import ai.djl.Application;
import ai.djl.Model;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.modality.cv.translator.ImageServingTranslator;
import ai.djl.util.ClassLoaderUtils;
import ai.djl.util.Pair;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Constructor;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;
import java.util.Set;

/** A {@link TranslatorFactory} that creates a generic {@link Translator}. */
public class ServingTranslatorFactory implements TranslatorFactory {

    private static final Logger logger = LoggerFactory.getLogger(ServingTranslatorFactory.class);

    /** {@inheritDoc} */
    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        return Collections.singleton(new Pair<>(Input.class, Output.class));
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public <I, O> Translator<I, O> newInstance(
            Class<I> input, Class<O> output, Model model, Map<String, ?> arguments)
            throws TranslateException {
        if (!isSupported(input, output)) {
            throw new IllegalArgumentException("Unsupported input/output types.");
        }

        Path modelDir = model.getModelPath();
        String factoryClass = ArgumentsUtil.stringValue(arguments, "translatorFactory");
        if (factoryClass != null && !factoryClass.isEmpty()) {
            TranslatorFactory factory = loadTranslatorFactory(factoryClass);
            if (factory != null && factory.isSupported(input, output)) {
                logger.info("Using TranslatorFactory: {}", factory.getClass().getName());
                return factory.newInstance(input, output, model, arguments);
            }
        }

        String className = (String) arguments.get("translator");

        Path libPath = modelDir.resolve("libs");
        if (!Files.isDirectory(libPath)) {
            libPath = modelDir.resolve("lib");
            if (!Files.isDirectory(libPath) && className == null) {
                return (Translator<I, O>) loadDefaultTranslator(arguments);
            }
        }
        ServingTranslator translator = findTranslator(libPath, className);
        if (translator != null) {
            translator.setArguments(arguments);
            logger.info("Using translator: {}", translator.getClass().getName());
            return (Translator<I, O>) translator;
        } else if (className != null) {
            throw new TranslateException("Failed to load translator: " + className);
        }
        return (Translator<I, O>) loadDefaultTranslator(arguments);
    }

    private ServingTranslator findTranslator(Path path, String className) {
        Path classesDir = path.resolve("classes");
        ClassLoaderUtils.compileJavaClass(classesDir);
        return ClassLoaderUtils.findImplementation(path, ServingTranslator.class, className);
    }

    private TranslatorFactory loadTranslatorFactory(String className) {
        try {
            Class<?> clazz = Class.forName(className);
            Class<? extends TranslatorFactory> subclass = clazz.asSubclass(TranslatorFactory.class);
            Constructor<? extends TranslatorFactory> constructor = subclass.getConstructor();
            return constructor.newInstance();
        } catch (Throwable e) {
            logger.trace("Not able to load TranslatorFactory: " + className, e);
        }
        return null;
    }

    private Translator<Input, Output> loadDefaultTranslator(Map<String, ?> arguments) {
        String appName = ArgumentsUtil.stringValue(arguments, "application");
        if (appName != null) {
            Application application = Application.of(appName);
            if (application == Application.CV.IMAGE_CLASSIFICATION) {
                return getImageClassificationTranslator(arguments);
            }
        }
        NoopServingTranslatorFactory factory = new NoopServingTranslatorFactory();
        return factory.newInstance(Input.class, Output.class, null, arguments);
    }

    private Translator<Input, Output> getImageClassificationTranslator(Map<String, ?> arguments) {
        return new ImageServingTranslator(ImageClassificationTranslator.builder(arguments).build());
    }
}
