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

import ai.djl.mxnet.engine.MxEngine;
import ai.djl.mxnet.zoo.cv.actionrecognition.ActionRecognitionModelLoader;
import ai.djl.mxnet.zoo.cv.classification.MlpModelLoader;
import ai.djl.mxnet.zoo.cv.classification.Resnest;
import ai.djl.mxnet.zoo.cv.classification.Resnet;
import ai.djl.mxnet.zoo.cv.classification.Resnext;
import ai.djl.mxnet.zoo.cv.classification.SeResnext;
import ai.djl.mxnet.zoo.cv.classification.Senet;
import ai.djl.mxnet.zoo.cv.classification.Squeezenet;
import ai.djl.mxnet.zoo.cv.objectdetection.SingleShotDetectionModelLoader;
import ai.djl.mxnet.zoo.cv.objectdetection.YoloModelLoader;
import ai.djl.mxnet.zoo.cv.poseestimation.SimplePoseModelLoader;
import ai.djl.mxnet.zoo.cv.segmentation.InstanceSegmentationModelLoader;
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
    private static final Repository REPOSITORY = Repository.newInstance("MxNet", DJL_REPO_URL);
    public static final String GROUP_ID = "ai.djl.mxnet";

    public static final MlpModelLoader MLP = new MlpModelLoader(REPOSITORY);
    public static final SingleShotDetectionModelLoader SSD =
            new SingleShotDetectionModelLoader(REPOSITORY);
    public static final YoloModelLoader YOLO = new YoloModelLoader(REPOSITORY);
    public static final Resnet RESNET = new Resnet(REPOSITORY);
    public static final Resnest RESNEST = new Resnest(REPOSITORY);
    public static final Resnext RESNEXT = new Resnext(REPOSITORY);
    public static final Senet SENET = new Senet(REPOSITORY);
    public static final SeResnext SE_RESNEXT = new SeResnext(REPOSITORY);
    public static final Squeezenet SQUEEZENET = new Squeezenet(REPOSITORY);
    public static final SimplePoseModelLoader SIMPLE_POSE = new SimplePoseModelLoader(REPOSITORY);
    public static final InstanceSegmentationModelLoader MASK_RCNN =
            new InstanceSegmentationModelLoader(REPOSITORY);
    public static final ActionRecognitionModelLoader ACTION_RECOGNITION =
            new ActionRecognitionModelLoader(REPOSITORY);
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
