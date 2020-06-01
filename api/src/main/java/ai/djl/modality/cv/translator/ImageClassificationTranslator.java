/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.List;

/** A generic {@link ai.djl.translate.Translator} for Image Classification tasks. */
public class ImageClassificationTranslator extends BaseImageTranslator<Classifications> {

    private SynsetLoader synsetLoader;
    private boolean applySoftmax;

    private List<String> synset;

    /**
     * Constructs an Image Classification using {@link Builder}.
     *
     * @param builder the data to build with
     */
    public ImageClassificationTranslator(Builder builder) {
        super(builder);
        this.synsetLoader = builder.synsetLoader;
        this.applySoftmax = builder.applySoftmax;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(NDManager manager, Model model) throws IOException {
        synset = synsetLoader.load(model);
    }

    /** {@inheritDoc} */
    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        NDArray probabilitiesNd = list.singletonOrThrow();
        if (applySoftmax) {
            probabilitiesNd = probabilitiesNd.softmax(0);
        }
        return new Classifications(synset, probabilitiesNd);
    }

    /**
     * Creates a builder to build a {@code ImageClassificationTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** A Builder to construct a {@code ImageClassificationTranslator}. */
    public static class Builder extends BaseBuilder<Builder> {

        private SynsetLoader synsetLoader;
        private boolean applySoftmax;

        Builder() {}

        /**
         * Sets the name of the synset file listing the potential classes for an image.
         *
         * @param synsetArtifactName a file listing the potential classes for an image
         * @return the builder
         */
        public Builder optSynsetArtifactName(String synsetArtifactName) {
            synsetLoader = new SynsetLoader(synsetArtifactName);
            return this;
        }

        /**
         * Sets the URL of the synset file.
         *
         * @param synsetUrl the URL of the synset file
         * @return the builder
         */
        public Builder optSynsetUrl(URL synsetUrl) {
            this.synsetLoader = new SynsetLoader(synsetUrl);
            return this;
        }

        /**
         * Sets the potential classes for an image.
         *
         * @param synset the potential classes for an image
         * @return the builder
         */
        public Builder optSynset(List<String> synset) {
            synsetLoader = new SynsetLoader(synset);
            return this;
        }

        /**
         * Sets whether to apply softmax when processing output. Some models already include softmax
         * in the last layer, so don't apply softmax when processing model output.
         *
         * @param applySoftmax boolean whether to apply softmax
         * @return the builder
         */
        public Builder optApplySoftmax(boolean applySoftmax) {
            this.applySoftmax = applySoftmax;
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds the {@link ImageClassificationTranslator} with the provided data.
         *
         * @return an {@link ImageClassificationTranslator}
         */
        public ImageClassificationTranslator build() {
            if (synsetLoader == null) {
                synsetLoader = new SynsetLoader("synset.txt");
            }
            return new ImageClassificationTranslator(this);
        }
    }

    private static final class SynsetLoader {

        private String synsetFileName;
        private URL synsetUrl;
        private List<String> synset;

        public SynsetLoader(List<String> synset) {
            this.synset = synset;
        }

        public SynsetLoader(URL synsetUrl) {
            this.synsetUrl = synsetUrl;
        }

        public SynsetLoader(String synsetFileName) {
            this.synsetFileName = synsetFileName;
        }

        public List<String> load(Model model) throws IOException {
            if (synset != null) {
                return synset;
            } else if (synsetUrl != null) {
                try (InputStream is = synsetUrl.openStream()) {
                    return Utils.readLines(is);
                }
            }
            return model.getArtifact(synsetFileName, Utils::readLines);
        }
    }
}
