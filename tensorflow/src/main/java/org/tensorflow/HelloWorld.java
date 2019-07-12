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

package org.tensorflow;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import javax.imageio.ImageIO;
import org.tensorflow.engine.TfModel;
import org.tensorflow.engine.TfNDArray;
import software.amazon.ai.TranslateException;
import software.amazon.ai.Translator;
import software.amazon.ai.TranslatorContext;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.Pair;

public final class HelloWorld {

    private HelloWorld() {}

    @SuppressWarnings("PMD.SystemPrintln")
    public static void main(String[] args) throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray a =
                    new TfNDArray(
                            manager, new Shape(1, 3), FloatBuffer.wrap(new float[] {1f, 2f, 3f}));

            System.out.println("Input Shape:");
            System.out.println(a.getShape());

            System.out.println("Softmax:");
            System.out.println(Arrays.toString(a.softmax().toFloatArray()));

            System.out.println("ZerosLike:");
            System.out.println(Arrays.toString(a.zerosLike().toFloatArray()));

            System.out.println("OnesLike:");
            System.out.println(Arrays.toString(a.onesLike().toFloatArray()));

            TfModel model = TfModel.loadModel("ModelPath/TF-resnet_ssd");
            System.out.println(model.describeInput()[0].getShape());
            System.out.println(model.describeInput()[0].getName());

            String filename = "ModelPath/TF-resnet_ssd/mfc.jpg";
            BufferedImage img = ImageIO.read(new File(filename));
            GenericTranslator translator = new GenericTranslator();
            Predictor<BufferedImage, NDList> predictor = Predictor.newInstance(model, translator);
            NDList list = predictor.predict(img);

            for (Pair<String, NDArray> pair : list) {
                System.out.println(pair.getKey() + " " + pair.getValue().getShape().toString());
            }
        }
    }

    private static void bgr2rgb(byte[] data) {
        for (int i = 0; i < data.length; i += 3) {
            byte tmp = data[i];
            data[i] = data[i + 2];
            data[i + 2] = tmp;
        }
    }

    private static final class GenericTranslator implements Translator<BufferedImage, NDList> {

        private static final int BATCH_SIZE = 1;
        private static final int CHANNELS = 3;

        @Override
        public NDList processInput(TranslatorContext ctx, BufferedImage img)
                throws TranslateException {
            if (img.getType() != BufferedImage.TYPE_3BYTE_BGR) {
                throw new TranslateException("");
            }
            byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
            // ImageIO.read seems to produce BGR-encoded images, but the model expects RGB.
            bgr2rgb(data);
            long[] shape = new long[] {BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS};
            NDArray array = ctx.getNDManager().create(new Shape(shape), ByteBuffer.wrap(data));
            NDList ndList = new NDList();
            ndList.add("image_tensor", array);
            return ndList;
        }

        @Override
        public NDList processOutput(TranslatorContext ctx, NDList list) throws TranslateException {
            return list;
        }
    }
}
