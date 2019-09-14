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
package software.amazon.ai.translate;

import java.util.ArrayList;
import java.util.List;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.training.dataset.Batchifier;

/**
 * The {@code Translator} interface provides model pre-processing and postprocessing functionality.
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
 *         return new NDList(ctx.getNDManager().create(dataDesc, buffer));
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
 *
 * @param <I> input type
 * @param <O> output type
 */
public interface Translator<I, O> extends PreProcessor<I>, PostProcessor<O> {

    // Default to Stack batchifier
    default Batchifier getBatchifier() {
        return Batchifier.STACK;
    }

    /**
     * Processes the inputs and converts it to batched NDList.
     *
     * @param ctx Toolkit that would help to creating input NDArray
     * @param inputs Input Objects
     * @return {@link NDList}
     * @throws Exception if an error occurs during processing input
     */
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    default NDList processInputBatch(TranslatorContext ctx, List<I> inputs) throws Exception {
        int batchSize = inputs.size();
        NDList[] preprocessed = new NDList[batchSize];
        int index = 0;
        for (I inp : inputs) {
            preprocessed[index++] = processInput(ctx, inp);
        }
        return getBatchifier().batchify(preprocessed);
    }

    /**
     * Processes the output batched NDList to the corresponding Output Objects.
     *
     * @param ctx Toolkit used to do postprocessing
     * @param list Batched Output NDList after inference
     * @return output objects
     * @throws Exception if an error occurs during processing output
     */
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    default List<O> processOutputBatch(TranslatorContext ctx, NDList list) throws Exception {
        NDList[] unbatched = getBatchifier().unbatchify(list);
        List<O> outputs = new ArrayList<>(unbatched.length);
        for (NDList output : unbatched) {
            outputs.add(processOutput(ctx, output));
        }
        return outputs;
    }
}
