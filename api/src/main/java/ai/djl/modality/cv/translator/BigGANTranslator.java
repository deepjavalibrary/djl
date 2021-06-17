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

import ai.djl.engine.Engine;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.input.BigGANInput;
import ai.djl.modality.cv.input.ImageNetCategory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Built-in {@code Translator} that provides preprocessing and postprocessing for BigGAN. */
public final class BigGANTranslator implements Translator<BigGANInput, Image[]> {
    private static final Logger logger = LoggerFactory.getLogger(BigGANTranslator.class);
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
        logOutputList(list);

        NDArray output = list.get(0).addi(1).muli(128).clip(0, 255).toType(DataType.UINT8, false);

        int sampleSize = (int) output.getShape().get(0);
        Image[] images = new Image[sampleSize];

        for (int i = 0; i < sampleSize; i++) {
            images[i] = ImageFactory.getInstance().fromNDArray(output.get(i));
        }

        return images;
    }

    private void logOutputList(NDList list) {
        logger.info("");
        logger.info("MY OUTPUT:");
        list.forEach(array -> logger.info("   out: {}", array.getShape()));
    }

    @Override
    public NDList processInput(TranslatorContext ctx, BigGANInput input) throws Exception {
        Engine.getInstance().setRandomSeed(0);
        NDManager manager = ctx.getNDManager();

        NDArray categoryArray = createCategoryArray(manager, input);
        NDArray seed =
                manager.truncatedNormal(new Shape(input.getSampleSize(), SEED_COLUMN_SIZE))
                        .muli(input.getTruncation());
        NDArray truncation = manager.create(input.getTruncation());

        logInputArrays(categoryArray, seed, truncation);
        return new NDList(seed, categoryArray, truncation);
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
        int categoryId = input.getCategory().getId();
        int sampleSize = input.getSampleSize();

        int[] indices = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            indices[i] = categoryId;
        }
        return manager.create(indices).oneHot(ImageNetCategory.NUMBER_OF_CATEGORIES);
    }

    private void logInputArrays(NDArray categoryArray, NDArray seed, NDArray truncation) {
        logger.info("");
        logger.info("MY INPUTS: ");
        logger.info("   y: {}", categoryArray.getShape());
        logger.info("   z: {}", seed.get(":, :10"));
        logger.info("   truncation: {}", truncation.getShape());
    }

    @Override
    public Batchifier getBatchifier() {
        return null;
    }
}
