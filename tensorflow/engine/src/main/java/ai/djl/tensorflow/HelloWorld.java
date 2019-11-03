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

package ai.djl.tensorflow;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.tensorflow.engine.TfModel;
import ai.djl.tensorflow.engine.TfNDArray;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Paths;
import java.util.Arrays;
import javax.imageio.ImageIO;

public final class HelloWorld {

    private HelloWorld() {}

    @SuppressWarnings("PMD.SystemPrintln")
    public static void main(String[] args)
            throws IOException, TranslateException, MalformedModelException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray a =
                    new TfNDArray(
                            manager, new Shape(1, 3), FloatBuffer.wrap(new float[] {1f, 2f, 3f}));

            System.out.println("Input Shape:");
            System.out.println(a.getShape());

            System.out.println("Softmax:");
            System.out.println(Arrays.toString(a.softmax(-1).toFloatArray()));

            System.out.println("ZerosLike:");
            System.out.println(Arrays.toString(a.zerosLike().toFloatArray()));

            System.out.println("OnesLike:");
            System.out.println(Arrays.toString(a.onesLike().toFloatArray()));

            TfModel model = new TfModel();
            model.load(Paths.get("ModelPath/TF-resnet_ssd"));
            System.out.println(model.describeInput().get(0).getValue());
            System.out.println(model.describeInput().get(0).getKey());

            String filename = "ModelPath/TF-resnet_ssd/mfc.jpg";
            BufferedImage img = ImageIO.read(new File(filename));
            GenericTranslator translator = new GenericTranslator();
            Predictor<BufferedImage, NDList> predictor = model.newPredictor(translator);
            NDList list = predictor.predict(img);

            for (NDArray array : list) {
                System.out.println(array.getName() + ": " + array.getShape().toString());
            }
        }
    }

    private static final class GenericTranslator implements Translator<BufferedImage, NDList> {

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, BufferedImage img) {
            NDArray array = BufferedImageUtils.toNDArray(ctx.getNDManager(), img);
            array.setName("image_tensor");
            return new NDList(array);
        }

        /** {@inheritDoc} */
        @Override
        public NDList processOutput(TranslatorContext ctx, NDList list) {
            return list;
        }
    }
}
