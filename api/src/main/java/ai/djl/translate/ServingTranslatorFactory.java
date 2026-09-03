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
import ai.djl.util.ClassLoaderUtils;
import ai.djl.util.Pair;
import ai.djl.util.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Constructor;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Locale;
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
        if (factoryClass != null) {
            Translator<Input, Output> translator =
                    getServingTranslator(factoryClass, model, arguments);
            if (translator != null) {
                return (Translator<I, O>) translator;
            }
            throw new TranslateException("Failed to load translatorFactory: " + factoryClass);
        }

        String className = (String) arguments.get("translator");
        Path libPath = modelDir.resolve("libs");
        if (!Files.isDirectory(libPath)) {
            libPath = modelDir.resolve("lib");
            if (!Files.isDirectory(libPath) && className == null) {
                return (Translator<I, O>) loadDefaultTranslator(model, arguments);
            }
        }
        ServingTranslator servingTranslator = findTranslator(libPath, className);
        if (servingTranslator != null) {
            servingTranslator.setArguments(arguments);
            logger.info("Using translator: {}", servingTranslator.getClass().getName());
            return (Translator<I, O>) servingTranslator;
        } else if (className != null) {
            throw new TranslateException("Failed to load translator: " + className);
        }

        return (Translator<I, O>) loadDefaultTranslator(model, arguments);
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
            logger.trace("Not able to load TranslatorFactory: {}", className, e);
        }
        return null;
    }

    private Translator<Input, Output> loadDefaultTranslator(Model model, Map<String, ?> arguments)
            throws TranslateException {
        String factoryClass = detectTranslatorFactory(arguments);
        Translator<Input, Output> translator = getServingTranslator(factoryClass, model, arguments);
        if (translator != null) {
            return translator;
        }

        NoopServingTranslatorFactory factory = new NoopServingTranslatorFactory();
        return factory.newInstance(Input.class, Output.class, null, arguments);
    }

    private String detectTranslatorFactory(Map<String, ?> arguments) {
        Application application;
        String app = ArgumentsUtil.stringValue(arguments, "application");
        if (app != null) {
            application = Application.of(app);
        } else {
            String task = Utils.getEnvOrSystemProperty("HF_TASK");
            task = ArgumentsUtil.stringValue(arguments, "task", task);
            if (task != null) {
                task = task.replace("-", "_").toLowerCase(Locale.ROOT);
                application = Application.of(task);
            } else {
                application = Application.UNDEFINED;
            }
        }
        if (application == Application.CV.IMAGE_CLASSIFICATION) {
            return "ai.djl.modality.cv.translator.ImageClassificationTranslatorFactory";
        } else if (application == Application.NLP.FILL_MASK) {
            return "ai.djl.huggingface.translator.FillMaskTranslatorFactory";
        } else if (application == Application.NLP.QUESTION_ANSWER) {
            return "ai.djl.huggingface.translator.QuestionAnsweringTranslatorFactory";
        } else if (application == Application.NLP.TEXT_CLASSIFICATION) {
            return "ai.djl.huggingface.translator.TextClassificationTranslatorFactory";
        } else if (application == Application.NLP.TEXT_EMBEDDING) {
            return "ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory";
        } else if (application == Application.NLP.TOKEN_CLASSIFICATION) {
            return "ai.djl.huggingface.translator.TokenClassificationTranslatorFactory";
        }
        return null;
    }

    private Translator<Input, Output> getServingTranslator(
            String factoryClass, Model model, Map<String, ?> arguments) throws TranslateException {
        TranslatorFactory factory = loadTranslatorFactory(factoryClass);
        if (factory != null && factory.isSupported(Input.class, Output.class)) {
            logger.info("Using TranslatorFactory: {}", factoryClass);
            return factory.newInstance(Input.class, Output.class, model, arguments);
        }
        return null;
    }
}
