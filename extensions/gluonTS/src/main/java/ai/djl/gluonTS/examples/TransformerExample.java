/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.gluonTS.examples;

import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.gluonTS.ForeCast;
import ai.djl.gluonTS.GluonTSData;
import ai.djl.gluonTS.dataset.FieldName;
import ai.djl.gluonTS.translator.TransformerTranslator;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.HashMap;

/** The example is targeted to specific use case for Transformer time series forecast. */
public final class TransformerExample {

    private static final Logger logger = LoggerFactory.getLogger(TransformerExample.class);

    private TransformerExample() {}

    public static void main(String[] args) throws IOException, TranslateException, ModelException {
        logger.info(Engine.getInstance().getVersion());
        float[] results = TransformerExample.predict(args);
    }

    public static float[] predict(String[] args)
            throws IOException, TranslateException, ModelException {
        HashMap<String, Object> arguments = new HashMap<>();
        arguments.put("prediction_length", 28);
        TransformerTranslator.Builder builder = TransformerTranslator.builder(arguments);
        TransformerTranslator translator = builder.build();
        Criteria<GluonTSData, ForeCast> criteria =
                Criteria.builder()
                        .setTypes(GluonTSData.class, ForeCast.class)
                        .optModelPath(Paths.get("src/main/resources/trained model/transformer.tar"))
                        .optTranslator(translator)
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<GluonTSData, ForeCast> model = criteria.loadModel()) {
            try (Predictor<GluonTSData, ForeCast> predictor = model.newPredictor()) {
                GluonTSData input = new GluonTSData();
                input.setStartTime(LocalDateTime.parse("2011-01-29T00:00"));
                NDArray target = model.getNDManager().arange(1857f).toType(DataType.FLOAT32, false).div(1857f);
                input.setField(FieldName.TARGET, target);
                ForeCast foreCast = predictor.predict(input);
                NDArray median = foreCast.mean();
                float[] floats = median.toFloatArray();

                // [681. 491. 600. 353. 300. 412. 419. 158.  92.  97.  75.  34.  52.  69.
                //  37.  15.   7.  10.   7.   9.  12.  20.   6.   9.   7.   3.   3.  11.]
                logger.info(Arrays.toString(floats));
                return floats;
            }
        }
    }
}
