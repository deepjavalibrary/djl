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
import ai.djl.ndarray.NDList;

import java.util.ArrayList;
import java.util.List;

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
     * Returns the {@link Batchifier}.
     *
     * @return the {@link Batchifier}
     */
    default Batchifier getBatchifier() {
        return Batchifier.STACK;
    }

    /**
     * Batch processes the inputs and converts it to NDList.
     *
     * @param ctx the toolkit for creating the input NDArray
     * @param inputs a list of the input object
     * @return the {@link NDList} after pre-processing
     * @throws Exception if an error occurs during processing input
     */
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    default NDList batchProcessInput(TranslatorContext ctx, List<I> inputs) throws Exception {
        NDList[] preprocessed = new NDList[inputs.size()];
        int index = 0;
        for (I input : inputs) {
            preprocessed[index++] = processInput(ctx, input);
        }
        return getBatchifier().batchify(preprocessed);
    }

    /**
     * Batch processes the output NDList to the corresponding output objects.
     *
     * @param ctx the toolkit used for post-processing
     * @param list the output NDList after inference, usually immutable in engines like
     *     PyTorch. @see <a href="https://github.com/deepjavalibrary/djl/issues/1774">Issue 1774</a>
     * @return a list of the output object of expected type
     * @throws Exception if an error occurs during processing output
     */
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    default List<O> batchProcessOutput(TranslatorContext ctx, NDList list) throws Exception {
        NDList[] unbatched = getBatchifier().unbatchify(list);
        List<O> outputs = new ArrayList<>(unbatched.length);
        for (NDList output : unbatched) {
            outputs.add(processOutput(ctx, output));
        }
        return outputs;
    }

    /**
     * Prepares the translator with the manager and model to use.
     *
     * @param ctx the context for the {@code Predictor}.
     * @throws Exception if there is an error for preparing the translator
     */
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    default void prepare(TranslatorContext ctx) throws Exception {}

    /**
     * Returns possible {@link TranslatorOptions} that can be built using this {@link Translator}.
     *
     * @return possible options or null if not defined
     */
    default TranslatorOptions getExpansions() {
        return null;
    }
}
