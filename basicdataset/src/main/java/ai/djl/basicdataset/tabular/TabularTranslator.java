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
import ai.djl.basicdataset.tabular.TabularResults.TabularResult;
import ai.djl.basicdataset.tabular.utils.DynamicBuffer;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.basicdataset.tabular.utils.Featurizer;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.TranslatorOptions;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/** A {@link Translator} that can be used for {@link ai.djl.Application.Tabular} tasks. */
public class TabularTranslator implements Translator<ListFeatures, TabularResults> {

    private List<Feature> features;
    private List<Feature> labels;

    /**
     * Constructs a {@link TabularTranslator} with the given features and labels.
     *
     * @param features the features for inputs
     * @param labels the labels for outputs
     */
    public TabularTranslator(List<Feature> features, List<Feature> labels) {
        this.features = features;
        this.labels = labels;
    }

    /**
     * Constructs a tabular translator for a model.
     *
     * @param model the model
     * @param arguments the arguments to build the translator with
     */
    @SuppressWarnings("PMD.UnusedFormalParameter") // TODO: Remove when implementing function
    public TabularTranslator(Model model, Map<String, ?> arguments) {
        throw new UnsupportedOperationException(
                "Constructing the TabularTranslator from arguments is not currently supported");
    }

    /** {@inheritDoc} */
    @Override
    public TabularResults processOutput(TranslatorContext ctx, NDList list) throws Exception {
        List<TabularResult> results = new ArrayList<>(labels.size());
        float[] data = list.singletonOrThrow().toFloatArray();
        int dataIndex = 0;
        for (Feature label : labels) {
            Featurizer featurizer = label.getFeaturizer();
            int dataRequired = featurizer.dataRequired();
            Object deFeaturized =
                    featurizer.deFeaturize(
                            Arrays.copyOfRange(data, dataIndex, dataIndex + dataRequired));
            results.add(new TabularResult(label.getName(), deFeaturized));
            dataIndex += dataRequired;
        }
        return new TabularResults(results);
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, ListFeatures input) throws Exception {
        if (input.size() != features.size()) {
            throw new IllegalArgumentException(
                    "The TabularTranslator expects "
                            + features.size()
                            + " arguments but received "
                            + input.size());
        }

        DynamicBuffer bb = new DynamicBuffer();
        for (int i = 0; i < features.size(); i++) {
            String value = input.get(i);
            features.get(i).getFeaturizer().featurize(bb, value);
        }
        FloatBuffer buf = bb.getBuffer();
        return new NDList(ctx.getNDManager().create(buf, new Shape(bb.getLength())));
    }

    /** {@inheritDoc} */
    @Override
    public TranslatorOptions getExpansions() {
        return new TabularTranslatorFactory().withTranslator(this);
    }

    /**
     * Returns the features for the translator.
     *
     * @return the features for the translator
     */
    public List<Feature> getFeatures() {
        return features;
    }

    /**
     * Returns the labels for the translator.
     *
     * @return the labels for the translator
     */
    public List<Feature> getLabels() {
        return labels;
    }
}
