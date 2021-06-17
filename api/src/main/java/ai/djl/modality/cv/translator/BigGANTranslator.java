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
package ai.djl.modality.cv.translator;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.input.BigGANInput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.util.Arrays;

/** Built-in {@code Translator} that provides preprocessing and postprocessing for BigGAN. */
public final class BigGANTranslator implements Translator<BigGANInput, Image[]> {

    private static final int NUMBER_OF_CATEGORIES = 1000;
    private static final int SEED_COLUMN_SIZE = 128;

    /**
     * Creates the array of images generated.
     *
     * @param ctx the toolkit used for post-processing
     * @param list the output NDList after inference
     * @return the array of generated images
     */
    @Override
    public Image[] processOutput(TranslatorContext ctx, NDList list) {
        NDArray output = list.get(0).addi(1).muli(128).clip(0, 255).toType(DataType.UINT8, false);

        int sampleSize = (int) output.getShape().get(0);
        Image[] images = new Image[sampleSize];

        for (int i = 0; i < sampleSize; ++i) {
            images[i] = ImageFactory.getInstance().fromNDArray(output.get(i));
        }

        return images;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, BigGANInput input) throws Exception {
        NDManager manager = ctx.getNDManager();

        NDArray categoryArray = createCategoryArray(manager, input);
        NDArray seed =
                manager.truncatedNormal(new Shape(input.getSampleSize(), SEED_COLUMN_SIZE))
                        .muli(input.getTruncation());
        NDArray truncation = manager.create(input.getTruncation());
        return new NDList(seed, categoryArray, truncation);
    }

    @Override
    public Batchifier getBatchifier() {
        return null;
    }

    /**
     * Creates a one-hot matrix where each row is a one-hot vector indicating the chosen category to
     * sample.
     *
     * @param manager to create NDArrays
     * @param input the input object to pre-process
     * @return one-hot matrix
     */
    private NDArray createCategoryArray(NDManager manager, BigGANInput input) {
        int categoryId = input.getCategoryId();
        int sampleSize = input.getSampleSize();

        int[] indices = new int[sampleSize];
        Arrays.fill(indices, categoryId);
        return manager.create(indices).oneHot(NUMBER_OF_CATEGORIES);
    }
}
