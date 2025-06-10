/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.inference.nlp;

import ai.djl.ModelException;
import ai.djl.huggingface.translator.ZeroShotClassificationTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.translator.ZeroShotClassificationInput;
import ai.djl.modality.nlp.translator.ZeroShotClassificationOutput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public final class ZeroShotClassification {

    private static final Logger logger = LoggerFactory.getLogger(ZeroShotClassification.class);

    private ZeroShotClassification() {}

    public static void main(String[] args) throws ModelException, IOException, TranslateException {
        ZeroShotClassificationOutput ret = predict(false);
        logger.info("{}", JsonUtils.GSON_PRETTY.toJson(ret));

        ret = predict(true);
        logger.info("{}", JsonUtils.GSON_PRETTY.toJson(ret));
    }

    public static ZeroShotClassificationOutput predict(boolean multiLabels)
            throws ModelException, IOException, TranslateException {
        Criteria<ZeroShotClassificationInput, ZeroShotClassificationOutput> criteria =
                Criteria.builder()
                        .setTypes(
                                ZeroShotClassificationInput.class,
                                ZeroShotClassificationOutput.class)
                        .optModelUrls("djl://ai.djl.huggingface.pytorch/MoritzLaurer/ModernBERT-large-zeroshot-v2.0")
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new ZeroShotClassificationTranslatorFactory())
                        .optProgress(new ProgressBar())
                        .build();
        String prompt = "one day I will see the world";
        String[] candidates = {"travel", "cooking", "dancing", "exploration"};

        try (ZooModel<ZeroShotClassificationInput, ZeroShotClassificationOutput> model =
                        criteria.loadModel();
                Predictor<ZeroShotClassificationInput, ZeroShotClassificationOutput> predictor =
                        model.newPredictor()) {
            ZeroShotClassificationInput input =
                    new ZeroShotClassificationInput(prompt, candidates, multiLabels);
            return predictor.predict(input);
        }
    }
}
