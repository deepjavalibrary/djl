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
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

/**
 * A {@link TranslatorFactory} based on a {@link Translator} and it's {@link TranslatorOptions}.
 *
 * @param <IbaseT> the input type for the base translator
 * @param <ObaseT> the output type for the base translator
 */
@SuppressWarnings({"PMD.GenericsNaming", "InterfaceTypeParameterName"})
public abstract class ExpansionTranslatorFactory<IbaseT, ObaseT> implements TranslatorFactory {

    /** {@inheritDoc} */
    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        Set<Pair<Type, Type>> results = new HashSet<>();
        results.addAll(getExpansions().keySet());

        Set<Type> preProcessorTypes = new HashSet<>();
        preProcessorTypes.addAll(getPreprocessorExpansions().keySet());
        preProcessorTypes.add(getBaseInputType());

        Set<Type> postProcessorTypes = new HashSet<>();
        postProcessorTypes.addAll(getPostprocessorExpansions().keySet());
        postProcessorTypes.add(getBaseOutputType());

        for (Type i : preProcessorTypes) {
            for (Type o : postProcessorTypes) {
                results.add(new Pair<>(i, o));
            }
        }
        return results;
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Translator<I, O> newInstance(
            Class<I> input, Class<O> output, Model model, Map<String, ?> arguments) {
        Translator<IbaseT, ObaseT> baseTranslator = buildBaseTranslator(model, arguments);
        return newInstance(input, output, baseTranslator);
    }

    /**
     * Returns a new instance of the {@link Translator} class.
     *
     * @param <I> the input data type
     * @param <O> the output data type
     * @param input the input class
     * @param output the output class
     * @param translator the base translator to expand from
     * @return a new instance of the {@code Translator} class
     */
    @SuppressWarnings("unchecked")
    <I, O> Translator<I, O> newInstance(
            Class<I> input, Class<O> output, Translator<IbaseT, ObaseT> translator) {

        if (input.equals(getBaseInputType()) && output.equals(getBaseOutputType())) {
            return (Translator<I, O>) translator;
        }

        TranslatorExpansion<IbaseT, ObaseT> expansion =
                getExpansions().get(new Pair<>(input, output));
        if (expansion != null) {
            return (Translator<I, O>) expansion.apply(translator);
        }

        // Note that regular expansions take precedence over pre-processor+post-processor expansions
        PreProcessor<I> preProcessor = null;
        if (input.equals(getBaseInputType())) {
            preProcessor = (PreProcessor<I>) translator;
        } else {
            Function<PreProcessor<IbaseT>, PreProcessor<?>> expander =
                    getPreprocessorExpansions().get(input);
            if (expander != null) {
                preProcessor = (PreProcessor<I>) expander.apply(translator);
            }
        }

        PostProcessor<O> postProcessor = null;
        if (output.equals(getBaseOutputType())) {
            postProcessor = (PostProcessor<O>) translator;
        } else {
            Function<PostProcessor<ObaseT>, PostProcessor<?>> expander =
                    getPostprocessorExpansions().get(output);
            if (expander != null) {
                postProcessor = (PostProcessor<O>) expander.apply(translator);
            }
        }

        if (preProcessor != null && postProcessor != null) {
            return new BasicTranslator<>(preProcessor, postProcessor, translator.getBatchifier());
        }

        throw new IllegalArgumentException("Unsupported expansion input/output types.");
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
     * Returns the input type for the base translator.
     *
     * @return the input type for the base translator
     */
    public abstract Class<IbaseT> getBaseInputType();

    /**
     * Returns the output type for the base translator.
     *
     * @return the output type for the base translator
     */
    public abstract Class<ObaseT> getBaseOutputType();

    /**
     * Returns the possible expansions of this factory.
     *
     * @return the possible expansions of this factory
     */
    protected Map<Pair<Type, Type>, TranslatorExpansion<IbaseT, ObaseT>> getExpansions() {
        return Collections.emptyMap();
    }

    /**
     * Returns the possible expansions of this factory.
     *
     * @return the possible expansions of this factory
     */
    protected Map<Type, Function<PreProcessor<IbaseT>, PreProcessor<?>>>
            getPreprocessorExpansions() {
        return Collections.singletonMap(getBaseInputType(), p -> p);
    }

    /**
     * Returns the possible expansions of this factory.
     *
     * @return the possible expansions of this factory
     */
    protected Map<Type, Function<PostProcessor<ObaseT>, PostProcessor<?>>>
            getPostprocessorExpansions() {
        return Collections.singletonMap(getBaseOutputType(), p -> p);
    }

    /** Represents {@link TranslatorOptions} by applying expansions to a base {@link Translator}. */
    final class ExpandedTranslatorOptions implements TranslatorOptions {

        private Translator<IbaseT, ObaseT> translator;

        private ExpandedTranslatorOptions(Translator<IbaseT, ObaseT> translator) {
            this.translator = translator;
        }

        /** {@inheritDoc} */
        @Override
        public Set<Pair<Type, Type>> getOptions() {
            return getSupportedTypes();
        }

        /** {@inheritDoc} */
        @Override
        public <I, O> Translator<I, O> option(Class<I> input, Class<O> output) {
            return newInstance(input, output, translator);
        }
    }

    /**
     * A function from a base translator to an expanded translator.
     *
     * @param <IbaseT> the base translator input type
     * @param <ObaseT> the base translator output type
     */
    @FunctionalInterface
    @SuppressWarnings({"PMD.GenericsNaming", "InterfaceTypeParameterName"})
    public interface TranslatorExpansion<IbaseT, ObaseT>
            extends Function<Translator<IbaseT, ObaseT>, Translator<?, ?>> {}
}
