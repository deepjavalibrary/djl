/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.basicdataset.tabular;

import ai.djl.Model;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDList;
import ai.djl.translate.ExpansionTranslatorFactory;
import ai.djl.translate.PostProcessor;
import ai.djl.translate.PreProcessor;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.lang.reflect.Type;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/** A {@link ai.djl.translate.TranslatorFactory} to extend the {@link TabularTranslator}. */
public class TabularTranslatorFactory
        extends ExpansionTranslatorFactory<ListFeatures, TabularResults> {

    /** {@inheritDoc} */
    @Override
    protected Translator<ListFeatures, TabularResults> buildBaseTranslator(
            Model model, Map<String, ?> arguments) {
        return new TabularTranslator(model, arguments);
    }

    /** {@inheritDoc} */
    @Override
    public Class<ListFeatures> getBaseInputType() {
        return ListFeatures.class;
    }

    /** {@inheritDoc} */
    @Override
    public Class<TabularResults> getBaseOutputType() {
        return TabularResults.class;
    }

    /** {@inheritDoc} */
    @Override
    protected Map<Type, Function<PreProcessor<ListFeatures>, PreProcessor<?>>>
            getPreprocessorExpansions() {
        Map<Type, Function<PreProcessor<ListFeatures>, PreProcessor<?>>> expansions =
                new ConcurrentHashMap<>();
        expansions.put(MapFeatures.class, MapPreProcessor::new);
        return expansions;
    }

    /** {@inheritDoc} */
    @Override
    protected Map<Type, Function<PostProcessor<TabularResults>, PostProcessor<?>>>
            getPostprocessorExpansions() {
        Map<Type, Function<PostProcessor<TabularResults>, PostProcessor<?>>> expansions =
                new ConcurrentHashMap<>();
        expansions.put(Classifications.class, ClassificationsTabularPostProcessor::new);
        expansions.put(Float.class, RegressionTabularPostProcessor::new);
        return expansions;
    }

    static class MapPreProcessor implements PreProcessor<MapFeatures> {

        private TabularTranslator preProcessor;

        MapPreProcessor(PreProcessor<ListFeatures> preProcessor) {
            if (!(preProcessor instanceof TabularTranslator)) {
                throw new IllegalArgumentException(
                        "The MapPreProcessor for the TabularTranslatorFactory expects a"
                                + " TabularTranslator, but received "
                                + preProcessor.getClass().getName());
            }
            this.preProcessor = (TabularTranslator) preProcessor;
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, MapFeatures input) throws Exception {
            ListFeatures list = new ListFeatures(preProcessor.getFeatures().size());
            for (Feature feature : preProcessor.getFeatures()) {
                if (input.containsKey(feature.getName())) {
                    list.add(input.get(feature.getName()));
                } else {
                    throw new IllegalArgumentException(
                            "The input to the TabularTranslator is missing the feature: "
                                    + feature.getName());
                }
            }
            return preProcessor.processInput(ctx, list);
        }
    }

    static class ClassificationsTabularPostProcessor implements PostProcessor<Classifications> {

        private PostProcessor<TabularResults> postProcessor;

        ClassificationsTabularPostProcessor(PostProcessor<TabularResults> postProcessor) {
            this.postProcessor = postProcessor;
        }

        /** {@inheritDoc} */
        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) throws Exception {
            TabularResults results = postProcessor.processOutput(ctx, list);

            if (results.size() != 1) {
                throw new IllegalStateException(
                        "The ClassificationsTabularPostProcessor expected the model to produce one"
                                + " output, but instead it produced "
                                + results.size());
            }

            Object result = results.getFeature(0).getResult();
            if (result instanceof Classifications) {
                return (Classifications) result;
            }
            throw new IllegalStateException(
                    "The ClassificationsTabularPostProcessor expected the model to produce a"
                            + " Classifications, but instead it produced "
                            + result.getClass().getName());
        }
    }

    static class RegressionTabularPostProcessor implements PostProcessor<Float> {

        private PostProcessor<TabularResults> postProcessor;

        RegressionTabularPostProcessor(PostProcessor<TabularResults> postProcessor) {
            this.postProcessor = postProcessor;
        }

        /** {@inheritDoc} */
        @Override
        public Float processOutput(TranslatorContext ctx, NDList list) throws Exception {
            TabularResults results = postProcessor.processOutput(ctx, list);

            if (results.size() != 1) {
                throw new IllegalStateException(
                        "The RegressionTabularPostProcessor expected the model to produce one"
                                + " output, but instead it produced "
                                + results.size());
            }

            Object result = results.getFeature(0).getResult();
            if (result instanceof Float) {
                return (Float) result;
            }
            throw new IllegalStateException(
                    "The RegressionTabularPostProcessor expected the model to produce a float, but"
                            + " instead it produced "
                            + result.getClass().getName());
        }
    }
}
