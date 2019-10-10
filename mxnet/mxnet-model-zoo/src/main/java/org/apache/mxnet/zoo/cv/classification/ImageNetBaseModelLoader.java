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
package org.apache.mxnet.zoo.cv.classification;

import java.awt.image.BufferedImage;
import org.apache.mxnet.zoo.BaseModelLoader;
import org.apache.mxnet.zoo.ModelZoo;
import software.amazon.ai.modality.Classification;
import software.amazon.ai.repository.Anchor;
import software.amazon.ai.repository.MRL;
import software.amazon.ai.repository.MRL.Model.CV;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.translate.Translator;

public abstract class ImageNetBaseModelLoader
        extends BaseModelLoader<BufferedImage, Classification> {

    private static final Anchor BASE_ANCHOR = CV.IMAGE_CLASSIFICATION;
    private static final String GROUP_ID = ModelZoo.GROUP_ID;

    public ImageNetBaseModelLoader(Repository repository, String artifactId, String version) {
        super(repository, new MRL(BASE_ANCHOR, GROUP_ID, artifactId), version);
    }

    @Override
    public Translator<BufferedImage, Classification> getTranslator() {
        return new ImageClassificationTranslator.Builder()
                .optCenterCrop()
                .optResize(224, 224)
                .setSynsetArtifactName("synset.txt")
                .build();
    }
}
