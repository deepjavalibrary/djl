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
package software.amazon.ai.example;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.TranslateException;
import software.amazon.ai.Translator;
import software.amazon.ai.TranslatorContext;
import software.amazon.ai.example.util.AbstractExample;
import software.amazon.ai.example.util.Arguments;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.modality.Classification;
import software.amazon.ai.modality.cv.Images;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.Shape;

public final class ClassifyExample extends AbstractExample {

    public static void main(String[] args) {
        new ClassifyExample().runExample(args);
    }

    @Override
    public Classification predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, TranslateException {
        Classification predictResult = null;
        Path modelDir = arguments.getModelDir();
        String modelName = arguments.getModelName();
        Path imageFile = arguments.getImageFile();
        BufferedImage img = Images.loadImageFromFile(imageFile);

        Model model = Model.loadModel(modelDir, modelName);

        ClassifyTranslator translator = new ClassifyTranslator(5, 224, 224);

        // Following context is not required, default context will be used by Predictor without
        // passing context to Predictor.newInstance(model, translator)
        // Change to a specific context if needed.
        Context context = Context.defaultContext();

        try (Predictor<BufferedImage, List<Classification>> predictor =
                Predictor.newInstance(model, translator, context)) {
            predictor.setMetrics(metrics); // Let predictor collect metrics

            for (int i = 0; i < iteration; ++i) {
                List<Classification> result = predictor.predict(img);
                predictResult = result.get(0);
                printProgress(iteration, i);
                collectMemoryInfo(metrics);
            }
        }

        model.close();
        return predictResult;
    }

    private static final class ClassifyTranslator
            implements Translator<BufferedImage, List<Classification>> {

        private int topK;
        private DataDesc dataDesc;
        private int imageWidth;
        private int imageHeight;

        public ClassifyTranslator(int topK, int imageWidth, int imageHeight) {
            this.topK = topK;
            this.imageWidth = imageWidth;
            this.imageHeight = imageHeight;
            dataDesc = new DataDesc(new Shape(1, 3, imageWidth, imageHeight), "data");
        }

        @Override
        public NDList processInput(TranslatorContext ctx, BufferedImage input) {
            BufferedImage image = Images.resizeImage(input, imageWidth, imageHeight);
            FloatBuffer buffer = Images.toFloatBuffer(image);

            return new NDList(ctx.getNDScopedFactory().create(dataDesc, buffer));
        }

        @Override
        public List<Classification> processOutput(TranslatorContext ctx, NDList list)
                throws IOException {
            Model model = ctx.getModel();
            NDArray array = list.get(0).get(0);

            long length = array.getShape().head();
            length = Math.min(length, topK);
            List<Classification> ret = new ArrayList<>(Math.toIntExact(length));
            NDArray sorted = array.argsort(-1, false);
            NDArray top = sorted.get(":" + topK);

            float[] probabilities = array.toFloatArray();
            int[] indices = top.toIntArray();

            String[] synset = model.getArtifact("synset.txt", AbstractExample::loadSynset);
            for (int i = 0; i < topK; ++i) {
                int index = indices[i];
                String className = synset[index];
                Classification out = new Classification(className, probabilities[index]);
                ret.add(out);
            }
            return ret;
        }
    }
}
