/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.inference.clip;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;

import java.io.IOException;

/**
 * An example of inference using an CLIP model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/clip_image_text.md">doc</a>
 * for information about this example.
 */
public class ClipModel {

    private Predictor<Image, float[]> imageFeatureExtractor;
    private Predictor<String, float[]> textFeatureExtractor;
    private Predictor<Pair<Image, String>, float[]> imgTextComparator;

    public ClipModel() throws ModelException, IOException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls("https://resources.djl.ai/demo/pytorch/clip.zip")
                        .optTranslator(new NoopTranslator())
                        .optEngine("PyTorch")
                        .build();
        ZooModel<NDList, NDList> clip = criteria.loadModel();
        imageFeatureExtractor = clip.newPredictor(new ImageTranslator());
        textFeatureExtractor = clip.newPredictor(new TextTranslator());
        imgTextComparator = clip.newPredictor(new ImageTextTranslator());
    }

    public float[] extractTextFeatures(String inputs) throws TranslateException {
        return textFeatureExtractor.predict(inputs);
    }

    public float[] extractImageFeatures(Image inputs) throws TranslateException {
        return imageFeatureExtractor.predict(inputs);
    }

    public float[] compareTextAndImage(Image image, String text) throws TranslateException {
        return imgTextComparator.predict(new Pair<>(image, text));
    }
}
