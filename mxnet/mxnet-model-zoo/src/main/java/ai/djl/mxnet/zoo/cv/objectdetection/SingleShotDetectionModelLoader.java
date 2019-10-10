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
package ai.djl.mxnet.zoo.cv.objectdetection;

import ai.djl.mxnet.zoo.BaseModelLoader;
import ai.djl.mxnet.zoo.ModelZoo;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import software.amazon.ai.modality.cv.DetectedObjects;
import software.amazon.ai.repository.Anchor;
import software.amazon.ai.repository.Artifact;
import software.amazon.ai.repository.MRL;
import software.amazon.ai.repository.MRL.Model.CV;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.translate.Translator;
import software.amazon.ai.zoo.ModelNotFoundException;

public class SingleShotDetectionModelLoader
        extends BaseModelLoader<BufferedImage, DetectedObjects> {

    private static final Anchor BASE_ANCHOR = CV.OBJECT_DETECTION;
    private static final String GROUP_ID = ModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "ssd";
    private static final String VERSION = "0.0.2";

    public SingleShotDetectionModelLoader(Repository repository) {
        super(repository, new MRL(BASE_ANCHOR, GROUP_ID, ARTIFACT_ID), VERSION);
    }

    @Override
    public Translator<BufferedImage, DetectedObjects> getTranslator() {
        return new SingleShotDetectionTranslator.Builder()
                .optResize(512, 512)
                .setSynsetArtifactName("classes.txt")
                .build();
    }

    @Override
    public Artifact match(Map<String, String> criteria) throws IOException, ModelNotFoundException {
        List<Artifact> list = search(criteria);
        if (list.isEmpty()) {
            return null;
        }

        list.sort(Artifact.COMPARATOR);
        return list.get(list.size() - 1);
    }
}
