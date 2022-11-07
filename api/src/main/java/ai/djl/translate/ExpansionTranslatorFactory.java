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
import ai.djl.util.Pair;

import java.lang.reflect.Type;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

/**
 * A {@link TranslatorFactory} based on a {@link Translator} and it's {@link TranslatorOptions}.
 *
 * @param <IbaseT> the input type for the base translator
 * @param <ObaseT> the output type for the base translator
 */
@SuppressWarnings("PMD.GenericsNaming")
public abstract class ExpansionTranslatorFactory<IbaseT, ObaseT> implements TranslatorFactory {

    /** {@inheritDoc} */
    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        return getExpansions().keySet();
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Translator<I, O> newInstance(
            Class<I> input, Class<O> output, Model model, Map<String, ?> arguments) {
        Translator<IbaseT, ObaseT> baseTranslator = buildBaseTranslator(model, arguments);
        return newInstance(input, output, baseTranslator);
    }

    /** {@inheritDoc} */
    @SuppressWarnings("unchecked")
    public <I, O> Translator<I, O> newInstance(
            Class<I> input, Class<O> output, Translator<IbaseT, ObaseT> translator) {
        Function<Translator<IbaseT, ObaseT>, Translator<?, ?>> expansion =
                getExpansions().get(new Pair<>(input, output));
        if (expansion == null) {
            throw new IllegalArgumentException("Unsupported expansion input/output types.");
        }
        return (Translator<I, O>) expansion.apply(translator);
    }

    /**
     * Creates a set of {@link TranslatorOptions} based on the expansions of a given translator.
     *
     * @param translator the translator to expand
     * @return the {@link TranslatorOptions}
     */
    public ExpandedTranslatorOptions withTranslator(Translator<IbaseT, ObaseT> translator) {
        return new ExpandedTranslatorOptions(translator);
    }

    /**
     * Builds the base translator that can be expanded.
     *
     * @param model the {@link Model} that uses the {@link Translator}
     * @param arguments the configurations for a new {@code Translator} instance
     * @return a base translator that can be expanded to form the factory options
     */
    protected abstract Translator<IbaseT, ObaseT> buildBaseTranslator(
            Model model, Map<String, ?> arguments);

    /**
     * Returns the possible expansions of this factory.
     *
     * @return the possible expansions of this factory
     */
    protected abstract Map<Pair<Type, Type>, Function<Translator<IbaseT, ObaseT>, Translator<?, ?>>>
            getExpansions();

    final class ExpandedTranslatorOptions implements TranslatorOptions {
        private Translator<IbaseT, ObaseT> translator;

        private ExpandedTranslatorOptions(Translator<IbaseT, ObaseT> translator) {
            this.translator = translator;
        }

        @Override
        public Set<Pair<Type, Type>> getOptions() {
            return getSupportedTypes();
        }

        @Override
        public <I, O> Translator<I, O> option(Class<I> input, Class<O> output) {
            return newInstance(input, output, translator);
        }
    }
}
