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
package ai.djl.mxnet.zoo;

import ai.djl.modality.cv.zoo.ActionRecognitionModelLoader;
import ai.djl.modality.cv.zoo.ImageClassificationModelLoader;
import ai.djl.modality.cv.zoo.InstanceSegmentationModelLoader;
import ai.djl.modality.cv.zoo.ObjectDetectionModelLoader;
import ai.djl.modality.cv.zoo.SimplePoseModelLoader;
import ai.djl.modality.cv.zoo.YoloModelLoader;
import ai.djl.mxnet.engine.MxEngine;
import ai.djl.mxnet.zoo.nlp.embedding.GloveWordEmbeddingModelLoader;
import ai.djl.mxnet.zoo.nlp.qa.BertQAModelLoader;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.ModelZoo;
import java.util.Collections;
import java.util.Set;

/**
 * MxModelZoo is a repository that contains all MXNet models in {@link
 * ai.djl.mxnet.engine.MxSymbolBlock} for DJL.
 */
public class MxModelZoo implements ModelZoo {

    private static final String DJL_REPO_URL = "https://mlrepo.djl.ai/";
    private static final Repository REPOSITORY = Repository.newInstance("MXNet", DJL_REPO_URL);
    public static final String GROUP_ID = "ai.djl.mxnet";
    private static final MxModelZoo ZOO = new MxModelZoo();

    public static final ObjectDetectionModelLoader SSD =
            new ObjectDetectionModelLoader(REPOSITORY, GROUP_ID, "ssd", "0.0.1", ZOO);
    public static final YoloModelLoader YOLO =
            new YoloModelLoader(REPOSITORY, GROUP_ID, "yolo", "0.0.1", ZOO);

    public static final ImageClassificationModelLoader ALEXNET =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "alexnet", "0.0.1", ZOO);
    public static final ImageClassificationModelLoader DARKNET =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "darknet", "0.0.1", ZOO);
    public static final ImageClassificationModelLoader DENSENET =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "densenet", "0.0.1", ZOO);
    public static final ImageClassificationModelLoader GOOGLENET =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "googlenet", "0.0.1", ZOO);
    public static final ImageClassificationModelLoader INCEPTIONV3 =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "inceptionv3", "0.0.1", ZOO);
    public static final ImageClassificationModelLoader MLP =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "mlp", "0.0.1", ZOO);
    public static final ImageClassificationModelLoader MOBILENET =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "mobilenet", "0.0.1", ZOO);
    public static final ImageClassificationModelLoader RESNEST =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "resnest", "0.0.1", ZOO);
    public static final ImageClassificationModelLoader RESNET =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "resnet", "0.0.1", ZOO);
    public static final ImageClassificationModelLoader RESNEXT =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "resnext", "0.0.1", ZOO);
    public static final ImageClassificationModelLoader SENET =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "senet", "0.0.1", ZOO);
    public static final ImageClassificationModelLoader SE_RESNEXT =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "se_resnext", "0.0.1", ZOO);
    public static final ImageClassificationModelLoader SQUEEZENET =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "squeezenet", "0.0.1", ZOO);
    public static final ImageClassificationModelLoader VGG =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "vgg", "0.0.1", ZOO);
    public static final ImageClassificationModelLoader XCEPTION =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "xception", "0.0.1", ZOO);

    public static final SimplePoseModelLoader SIMPLE_POSE =
            new SimplePoseModelLoader(REPOSITORY, GROUP_ID, "simple_pose", "0.0.1", ZOO);
    public static final InstanceSegmentationModelLoader MASK_RCNN =
            new InstanceSegmentationModelLoader(REPOSITORY, GROUP_ID, "mask_rcnn", "0.0.1", ZOO);
    public static final ActionRecognitionModelLoader ACTION_RECOGNITION =
            new ActionRecognitionModelLoader(
                    REPOSITORY, GROUP_ID, "action_recognition", "0.0.1", ZOO);
    public static final BertQAModelLoader BERT_QA = new BertQAModelLoader(REPOSITORY);
    public static final GloveWordEmbeddingModelLoader GLOVE =
            new GloveWordEmbeddingModelLoader(REPOSITORY);

    /** {@inheritDoc} */
    @Override
    public String getGroupId() {
        return GROUP_ID;
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedEngines() {
        return Collections.singleton(MxEngine.ENGINE_NAME);
    }
}
