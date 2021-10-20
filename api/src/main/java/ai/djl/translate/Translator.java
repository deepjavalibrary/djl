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
package ai.djl.translate;

import ai.djl.inference.Predictor;

/**
 * The {@code Translator} interface provides model pre-processing and postprocessing functionality.
 *
 * <p>Users can use this in {@link Predictor} with input and output objects specified. The
 * recommended flow is to use the Translator to translate only a single data item at a time ({@link
 * ai.djl.training.dataset.Record}) rather than a Batch. For example, the input parameter would then
 * be {@code Image} rather than {@code Image[]}. The {@link ai.djl.training.dataset.Record}s will
 * then be combined using a {@link Batchifier}. If it is easier in your use case to work with
 * batches directly or your model uses records instead of batches, you can use the {@link
 * NoBatchifyTranslator}.
 *
 * <p>The following is an example of processing an image and creating classification output:
 *
 * <pre>
 * private static final class MyTranslator implements Translator&lt;Image, Classification&gt; {
 *
 *     private int imageWidth;
 *     private int imageHeight;
 *
 *     public MyTranslator(int imageWidth, int imageHeight) {
 *         this.imageWidth = imageWidth;
 *         this.imageHeight = imageHeight;
 *     }
 *
 *     &#064;Override
 *     public NDList processInput(TranslatorContext ctx, Image input) {
 *         NDArray imageND = input.toNDArray(ctx.getNDManager());
 *         return new NDList(NDImageUtils.resize(imageND, imageWidth, imageHeight);
 *     }
 *
 *     &#064;Override
 *     public Classification processOutput(TranslatorContext ctx, NDList list) throws TranslateException {
 *         Model model = ctx.getModel();
 *         NDArray array = list.get(0).at(0);
 *         NDArray sorted = array.argSort(-1, false);
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
 *
 * @param <I> the input type
 * @param <O> the output type
 */
public interface Translator<I, O> extends PreProcessor<I>, PostProcessor<O> {

    /**
     * Gets the {@link Batchifier}.
     *
     * @return the {@link Batchifier}
     */
    default Batchifier getBatchifier() {
        return Batchifier.STACK;
    }

    /**
     * Prepares the translator with the manager and model to use.
     *
     * @param ctx the context for the {@code Predictor}.
     * @throws Exception if there is an error for preparing the translator
     */
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    default void prepare(TranslatorContext ctx) throws Exception {}
}
