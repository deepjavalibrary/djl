/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.Model;
import ai.djl.repository.zoo.Criteria;
import ai.djl.util.Pair;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Constructor;
import java.lang.reflect.Type;
import java.util.Collections;
import java.util.Map;
import java.util.Set;

/**
 * A {@link TranslatorFactory} that creates the {@link Translator} based on serving.properties file.
 *
 * <p>The {@link Criteria} API cannot access serving.properties files before it's downloaded. A
 * {@code DeferredTranslatorFactory} assumes serving.properties will provide proper {@link
 * Translator}. If no translatorFactory is provided in serving.properties, a {@link
 * TranslateException} will be thrown.
 */
public class DeferredTranslatorFactory implements TranslatorFactory {

    private static final Logger logger = LoggerFactory.getLogger(DeferredTranslatorFactory.class);

    /** {@inheritDoc} */
    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        return Collections.emptySet();
    }

    /** {@inheritDoc} */
    @Override
    public boolean isSupported(Class<?> input, Class<?> output) {
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public Translator<?, ?> newInstance(
            Class<?> input, Class<?> output, Model model, Map<String, ?> arguments)
            throws TranslateException {
        String factoryClass = ArgumentsUtil.stringValue(arguments, "translatorFactory");
        if (factoryClass == null || factoryClass.isEmpty()) {
            throw new TranslateException("No translatorFactory defined.");
        }

        TranslatorFactory factory = loadTranslatorFactory(factoryClass);
        if (factory == null) {
            throw new TranslateException("Failed to load translatorFactory: " + factoryClass);
        } else if (!factory.isSupported(input, output)) {
            throw new TranslateException(factoryClass + " doesn't support Input/Output.");
        }
        logger.info("Using TranslatorFactory: {}", factoryClass);
        return factory.newInstance(input, output, model, arguments);
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
}
