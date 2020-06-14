/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.inference;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.examples.inference.benchmark.util.Arguments;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import java.io.IOException;
import java.nio.file.Path;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class LoadModel {

    private static final Logger logger = LoggerFactory.getLogger(LoadModel.class);

    private LoadModel() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        Options options = Arguments.getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            Arguments arguments = new Arguments(cmd);

            Classifications classifications = predict(arguments);
            logger.info("{}", classifications);
        } catch (ParseException e) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.setLeftPadding(1);
            formatter.setWidth(120);
            formatter.printHelp(e.getMessage(), options);
        }
    }

    public static Classifications predict(Arguments arguments)
            throws IOException, ModelException, TranslateException {
        Path imageFile = arguments.getImageFile();
        Image img = ImageFactory.getInstance().fromFile(imageFile);
        String artifactId = arguments.getArtifactId();

        Criteria.Builder<Image, Classifications> builder =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(Image.class, Classifications.class)
                        .optArtifactId(artifactId)
                        .optFilters(arguments.getCriteria())
                        .optProgress(new ProgressBar());
        if (artifactId.startsWith("ai.djl.localmodelzoo")) {
            // load model from local folder
            // since local pre-trained model doesn't have a translator defined,
            // we need to supply a translator manually.
            builder.optTranslator(getTranslator());
        }

        Criteria<Image, Classifications> criteria = builder.build();
        try (ZooModel<Image, Classifications> model = ModelZoo.loadModel(criteria);
                Predictor<Image, Classifications> predictor = model.newPredictor()) {
            return predictor.predict(img);
        }
    }

    private static Translator<Image, Classifications> getTranslator() {
        // This ImageClassificationTranslator is just a default, you need to
        // make proper changes to match your local model's behavior.
        return ImageClassificationTranslator.builder()
                .addTransform(new CenterCrop())
                .addTransform(new Resize(224, 224))
                .addTransform(new ToTensor())
                .build();
    }
}
