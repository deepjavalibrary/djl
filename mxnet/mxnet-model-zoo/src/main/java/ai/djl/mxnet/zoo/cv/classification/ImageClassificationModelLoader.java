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
package ai.djl.mxnet.zoo.cv.classification;

import ai.djl.modality.Classification;
import ai.djl.modality.cv.ImageClassificationTranslator;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.repository.Anchor;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.MRL.Model.CV;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.translate.Translator;
import java.awt.image.BufferedImage;
import java.util.Map;

public abstract class ImageClassificationModelLoader
        extends BaseModelLoader<BufferedImage, Classification> {

    private static final Anchor BASE_ANCHOR = CV.IMAGE_CLASSIFICATION;
    private static final String GROUP_ID = MxModelZoo.GROUP_ID;

    public ImageClassificationModelLoader(
            Repository repository, String artifactId, String version) {
        super(repository, new MRL(BASE_ANCHOR, GROUP_ID, artifactId), version);
    }

    @Override
    public Translator<BufferedImage, Classification> getTranslator(Artifact artifact) {
        Map<String, Object> arguments = artifact.getArguments();
        int width = ((Double) arguments.getOrDefault("width", 224d)).intValue();
        int height = ((Double) arguments.getOrDefault("height", 224d)).intValue();

        return new ImageClassificationTranslator.Builder()
                .optCenterCrop()
                .optResize(height, width)
                .setSynsetArtifactName("synset.txt")
                .build();
    }
}
