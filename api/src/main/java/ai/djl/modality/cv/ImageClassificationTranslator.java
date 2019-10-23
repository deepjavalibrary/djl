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
import ai.djl.modality.Classification;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import java.io.IOException;
import java.util.List;

public class ImageClassificationTranslator extends ImageTranslator<Classification> {

    private String synsetArtifactName;

    public ImageClassificationTranslator(Builder builder) {
        super(builder);
        this.synsetArtifactName = builder.synsetArtifactName;
    }

    @Override
    public Classification processOutput(TranslatorContext ctx, NDList list) throws IOException {
        Model model = ctx.getModel();

        NDArray probabilitiesNd = list.singletonOrThrow().softmax(0);
        List<String> synset = model.getArtifact(synsetArtifactName, Utils::readLines);
        return new Classification(synset, probabilitiesNd);
    }

    public static class Builder extends BaseBuilder<Builder> {

        private String synsetArtifactName;

        public Builder setSynsetArtifactName(String synsetArtifactName) {
            this.synsetArtifactName = synsetArtifactName;
            return this;
        }

        @Override
        protected Builder self() {
            return this;
        }

        public ImageClassificationTranslator build() {
            if (synsetArtifactName == null) {
                throw new IllegalArgumentException("You must specify a synset artifact name");
            }
            return new ImageClassificationTranslator(this);
        }
    }
}
