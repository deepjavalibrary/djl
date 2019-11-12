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

package ai.djl.modality.cv;

import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import java.io.IOException;
import java.util.List;

/** A generic {@link ai.djl.translate.Translator} for Image Classification tasks. */
public class ImageClassificationTranslator extends ImageTranslator<Classifications> {

    private String synsetArtifactName;

    /**
     * Constructs an Image Classification using {@link Builder}.
     *
     * @param builder the data to build with
     */
    public ImageClassificationTranslator(Builder builder) {
        super(builder);
        this.synsetArtifactName = builder.synsetArtifactName;
    }

    /** {@inheritDoc} */
    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) throws IOException {
        Model model = ctx.getModel();

        NDArray probabilitiesNd = list.singletonOrThrow().softmax(0);
        List<String> synset = model.getArtifact(synsetArtifactName, Utils::readLines);
        return new Classifications(synset, probabilitiesNd);
    }

    /** A Builder to construct a {@link ImageClassificationTranslator}. */
    public static class Builder extends BaseBuilder<Builder> {

        private String synsetArtifactName;

        /**
         * Sets the name of the synset file listing the potential classes for an image.
         *
         * @param synsetArtifactName a file listing the potential classes for an image.
         * @return the builder
         */
        public Builder setSynsetArtifactName(String synsetArtifactName) {
            this.synsetArtifactName = synsetArtifactName;
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
         * @return an {@link ImageClassificationTranslator}.
         */
        public ImageClassificationTranslator build() {
            if (synsetArtifactName == null) {
                throw new IllegalArgumentException("You must specify a synset artifact name");
            }
            return new ImageClassificationTranslator(this);
        }
    }
}
