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
package software.amazon.ai;

import software.amazon.ai.ndarray.NDList;

/**
 * The <code>Translator</code> interface provides model pre-processing and postprocessing
 * functionality.
 *
 * <p>Users can use this in {@link software.amazon.ai.inference.Predictor} with input and output
 * objects specified. The following is an example of processing an image and creating classification
 * output:
 *
 * <pre>
 * private static final class MyTranslator implements Translator&lt;BufferedImage, Classification&gt; {
 *
 *     private DataDesc dataDesc;
 *     private int imageWidth;
 *     private int imageHeight;
 *
 *     public MyTranslator(int imageWidth, int imageHeight) {
 *         this.imageWidth = imageWidth;
 *         this.imageHeight = imageHeight;
 *         dataDesc = new DataDesc(new Shape(1, 3, imageWidth, imageHeight), "data");
 *     }
 *
 *     &#064;Override
 *     public NDList processInput(TranslatorContext ctx, BufferedImage input) {
 *         BufferedImage image = Images.resizeImage(input, imageWidth, imageHeight);
 *         FloatBuffer buffer = Images.toFloatBuffer(image);
 *
 *         return new NDList(ctx.getNDFactory().create(dataDesc, buffer));
 *     }
 *
 *     &#064;Override
 *     public Classification processOutput(TranslatorContext ctx, NDList list) throws TranslateException {
 *         Model model = ctx.getModel();
 *         NDArray array = list.get(0).at(0);
 *         NDArray sorted = array.argsort(-1, false);
 *         NDArray top = sorted.at(0);
 *
 *         float[] probabilities = array.toFloatArray();
 *         int topIndex = top.toIntArray()[0];
 *
 *         String[] synset;
 *         try {
 *             synset = model.getArtifact("synset.txt", MyTranslator::loadSynset);
 *         } catch (IOException e) {
 *             throw new TranslateException(e);
 *         }
 *         return new Classification(synset[topIndex], probabilities[topIndex]);
 *     }
 *
 *     private static String[] loadSynset(InputStream is) {
 *         ...
 *     }
 * }
 * </pre>
 */
public interface Translator<I, O> {

    /**
     * Processes the input and converts it to NDList.
     *
     * @param ctx Toolkit that would help to creating input NDArray
     * @param input Input Object
     * @return {@link NDList}
     * @throws TranslateException if an error occurs during processing input
     */
    NDList processInput(TranslatorContext ctx, I input) throws TranslateException;

    /**
     * Processes the output NDList to the corresponding Output Object.
     *
     * @param ctx Toolkit used to do postprocessing
     * @param list Output NDList after inference
     * @return output object
     * @throws TranslateException if an error occurs during processing output
     */
    O processOutput(TranslatorContext ctx, NDList list) throws TranslateException;
}
