/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.training.loss;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;

/**
 * {@code YOLOv3Loss} is an implementation of {@link Loss}. It is used to compute the loss while
 * training a YOLOv3 model for object detection. It involves computing the targets given the
 * generated anchors, labels and predictions, and then computing the sum of class predictions and
 * bounding box predictions.
 */
public final class YOLOv3Loss extends Loss {
    // TODO: currently not finished, still have some bugs inside and it can only be trained with
    // PyTorch Engine
    /*
       PRESETANCHORS shapes come from the K-means clustering of COCO dataset, which image size is 416*416
       it can be reshaped into any shape like 256*256, just multiply each value with 256/416
    */
    private static final float[] PRESETANCHORS = {
        116, 90, 156, 198, 373, 326,
        30, 61, 62, 45, 59, 119,
        10, 13, 16, 30, 33, 23
    };

    private float[] anchors;
    private int numClasses;
    private int boxAttr;
    private Shape inputShape;
    private float ignoreThreshold;
    private NDManager manager;
    private static final float EPSILON = 1e-7f;

    private YOLOv3Loss(Builder builder) {
        super(builder.name);
        this.anchors = builder.anchorsArray;
        this.numClasses = builder.numClasses;
        this.boxAttr = builder.numClasses + 5; // 5 for x,y,h,w,c
        this.inputShape = builder.inputShape;
        this.ignoreThreshold = builder.ignoreThreshold;
    }

    /**
     * Gets the preset anchors of YoloV3.
     *
     * @return the preset anchors of YoloV3
     */
    public static float[] getPresetAnchors() {
        return PRESETANCHORS.clone();
    }

    /**
     * Make the value of given NDArray between tMin and tMax.
     *
     * @param tList the given NDArray
     * @param tMin the min value
     * @param tMax the max value
     * @return a NDArray where values are set between tMin and tMax
     */
    public NDArray clipByTensor(NDArray tList, float tMin, float tMax) {
        NDArray result = tList.gte(tMin).mul(tList).add(tList.lt(tMin).mul(tMin));
        result = result.lte(tMax).mul(result).add(result.gt(tMax).mul(tMax));
        return result;
    }

    /**
     * Calculates the MSELoss between prediction and target.
     *
     * @param prediction the prediction array
     * @param target the target array
     * @return the MSELoss between prediction and target
     */
    public NDArray mseLoss(NDArray prediction, NDArray target) {
        return prediction.sub(target).pow(2);
    }

    /**
     * Calculates the BCELoss between prediction and target.
     *
     * @param prediction the prediction array
     * @param target the target array
     * @return the BCELoss between prediction and target
     */
    public NDArray bceLoss(NDArray prediction, NDArray target) {
        prediction = clipByTensor(prediction, EPSILON, (float) (1.0 - EPSILON));
        return prediction
                .log()
                .mul(target)
                .add(prediction.mul(-1).add(1).log().mul(target.mul(-1).add(1)))
                .mul(-1);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        manager = predictions.getManager();

        /*
        three outputs in total
        NDArray out0 = predictions.get(0),  //Shape = (batchSize * 75 * 13 * 13) 75 = 3*(20+5)
                out1 = predictions.get(1),  //Shape = (batchSize * 75 * 26 * 26)
                out2 = predictions.get(2);  //Shape = (batchSize * 75 * 52 * 52)
        */

        NDArray[] lossComponents = new NDArray[3];
        for (int i = 0; i < 3; i++) {
            lossComponents[i] = evaluateOneOutput(i, predictions.get(i), labels.singletonOrThrow());
        }

        // calculate the final loss
        return NDArrays.add(lossComponents);
    }

    /**
     * Computes the Loss for one outputLayer.
     *
     * @param componentIndex which outputLayer does current input represent. the shape should be
     *     (13*13,26*26,52*52)
     * @param input one prediction layer of YOLOv3
     * @param labels target labels. Must contain (offsetLabels, masks, classlabels)
     * @return the total loss of a outputLayer
     */
    public NDArray evaluateOneOutput(int componentIndex, NDArray input, NDArray labels) {
        int batchSize = (int) input.getShape().get(0);
        int inW = (int) input.getShape().get(2);
        int inH = (int) input.getShape().get(3);

        NDArray prediction =
                input.reshape(batchSize, 3, boxAttr, inW, inH)
                        .transpose(1, 0, 3, 4, 2); // reshape into (3,batchSize,inW,inH,attrs)

        // the prediction value of x,y,w,h which shape should be (3,batchSize,inW,inH)
        NDArray x = Activation.sigmoid(prediction.get("...,0"));
        NDArray y = Activation.sigmoid(prediction.get("...,1"));
        NDArray w = prediction.get("...,2");
        NDArray h = prediction.get("...,3");

        // Confidence of whether there is an object and conditional probability of each class
        // it should be reshaped into (batchSize,3)
        NDArray conf = Activation.sigmoid(prediction.get("...,4")).transpose(1, 0, 2, 3);
        NDArray predClass = Activation.sigmoid(prediction.get("...,5:")).transpose(1, 0, 2, 3, 4);

        // get an NDList of groundTruth which contains boxLossScale and groundTruth
        NDList truthList = getTarget(labels, inH, inW);

        /*
           boxLossScale shape should be: (batchSize,3,inW,inH)
           groundTruth shape should be: (3,batchSize,inW,inH,boxAttr)
        */
        NDArray boxLossScale = truthList.get(0).transpose(1, 0, 2, 3);
        NDArray groundTruth = truthList.get(1);

        // iou shape should be: (batchSize,3 ,inW,inH)
        NDArray iou =
                calculateIOU(x, y, groundTruth.get("...,0:4"), componentIndex)
                        .transpose(1, 0, 2, 3);

        // get noObjMask and objMask
        NDArray noObjMask =
                NDArrays.where(
                        iou.lte(ignoreThreshold), manager.ones(iou.getShape()), manager.create(0f));

        NDArray objMask = iou.argMax(1).oneHot(3).transpose(0, 3, 1, 2);

        objMask =
                NDArrays.where(
                        iou.gte(ignoreThreshold / 2),
                        objMask,
                        manager.zeros(objMask.getShape())); // to get rid of wrong ones
        noObjMask = NDArrays.where(objMask.eq(1f), manager.zeros(noObjMask.getShape()), noObjMask);

        NDArray xTrue = groundTruth.get("...,0");
        NDArray yTrue = groundTruth.get("...,1");
        NDArray wTrue = groundTruth.get("...,2");
        NDArray hTrue = groundTruth.get("...,3");
        NDArray classTrue = groundTruth.get("...,4:").transpose(1, 0, 2, 3, 4);

        NDArray widths =
                manager.create(
                                new float[] {
                                    anchors[componentIndex * 6],
                                    anchors[componentIndex * 6 + 2],
                                    anchors[componentIndex * 6 + 4]
                                })
                        .div(inputShape.get(0));

        NDArray heights =
                manager.create(
                                new float[] {
                                    anchors[componentIndex * 6 + 1],
                                    anchors[componentIndex * 6 + 3],
                                    anchors[componentIndex * 6 + 5]
                                })
                        .div(inputShape.get(1));

        // three loss parts: box Loss, confidence Loss, and class Loss
        NDArray boxLoss =
                objMask.mul(boxLossScale)
                        .mul(
                                NDArrays.add(
                                                xTrue.sub(x).pow(2),
                                                yTrue.sub(y).pow(2),
                                                wTrue.sub(
                                                                w.exp()
                                                                        .mul(
                                                                                widths.broadcast(
                                                                                                inH,
                                                                                                inW,
                                                                                                batchSize,
                                                                                                3)
                                                                                        .transpose(
                                                                                                3,
                                                                                                2,
                                                                                                1,
                                                                                                0)))
                                                        .pow(2),
                                                hTrue.sub(
                                                                h.exp()
                                                                        .mul(
                                                                                heights.broadcast(
                                                                                                inH,
                                                                                                inW,
                                                                                                batchSize,
                                                                                                3)
                                                                                        .transpose(
                                                                                                3,
                                                                                                2,
                                                                                                1,
                                                                                                0)))
                                                        .pow(2))
                                        .transpose(1, 0, 2, 3))
                        .sum();

        NDArray confLoss =
                objMask.mul(
                                conf.add(EPSILON)
                                        .log()
                                        .mul(-1)
                                        .add(bceLoss(predClass, classTrue).sum(new int[] {4})))
                        .sum();

        NDArray noObjLoss = noObjMask.mul(conf.mul(-1).add(1 + EPSILON).log().mul(-1)).sum();

        return boxLoss.add(confLoss).add(noObjLoss).div(batchSize);
    }

    /**
     * Gets target NDArray for a given evaluator.
     *
     * @param labels the true labels
     * @param inH the height of current layer
     * @param inW the width of current layer
     * @return an NDList of {boxLossScale and groundTruth}
     */
    public NDList getTarget(NDArray labels, int inH, int inW) {
        int batchSize = (int) labels.size(0);

        // the loss Scale of a box, used to punctuate small boxes
        NDList boxLossComponents = new NDList();

        // the groundTruth of a true object in pictures
        NDList groundTruthComponents = new NDList();

        // Shape of labels:(batchSize,objectNum,5)

        for (int batch = 0; batch < batchSize; batch++) {
            if (labels.get(batch).size(0) == 0) {
                continue; // no object in current picture
            }

            NDArray boxLoss = manager.zeros(new Shape(inW, inH), DataType.FLOAT32);
            NDArray groundTruth = manager.zeros(new Shape(inW, inH, boxAttr - 1), DataType.FLOAT32);

            NDArray picture = labels.get(batch);
            // the shape should be (objectNums,5)
            NDArray xgt = picture.get("...,1").add(picture.get("...,3").div(2)).mul(inW);
            // Center of x should be X value in labels and add half of the width and multiplies the
            // input width to get which grid cell it's in
            NDArray ygt = picture.get("...,2").add(picture.get("...,4").div(2)).mul(inH);
            // Center of y is the same as well
            NDArray wgt = picture.get("...,3");
            // the width of the ground truth box
            NDArray hgt = picture.get("...,4");
            // the height of the ground truth box
            // we should transform the presentation of true class, like
            // [[0],[1],[2]]->[[1,0,0,...0],[0,1,0,...,0],[0,0,1,...,0]]
            NDArray objectClass = picture.get("...,0");
            objectClass = objectClass.oneHot(numClasses);

            NDArray curLabel = labels.get(batch); // curLabel shape:(objectNum,5)
            int objectNum = (int) curLabel.size(0);

            for (int i = 0; i < objectNum; i++) {
                // for each object, the middle of the object(x and y) should be in one grid cell of
                // 13*13
                // the tx and ty should indicate the grid cell and bx and by should indicate the
                // movement from top-left of the grid cell
                int tx = (int) xgt.get(i).getFloat();
                int ty = (int) ygt.get(i).getFloat();
                float bx = xgt.get(i).getFloat() - tx;
                float by = ygt.get(i).getFloat() - ty;

                String index = tx + "," + ty;
                // set groundTruth
                groundTruth.set(new NDIndex(index + ",0"), bx);
                groundTruth.set(new NDIndex(index + ",1"), by);
                groundTruth.set(new NDIndex(index + ",2"), wgt.getFloat(i));
                groundTruth.set(new NDIndex(index + ",3"), hgt.getFloat(i));
                groundTruth.set(new NDIndex(index + ",4:"), objectClass.get(i));

                // set boxLoss
                boxLoss.set(new NDIndex(index), 2 - wgt.getFloat(i) * hgt.getFloat(i));
            }
            boxLossComponents.add(boxLoss);
            groundTruthComponents.add(groundTruth);
        }

        NDArray boxLossScale = NDArrays.stack(boxLossComponents).broadcast(3, batchSize, inW, inH);
        NDArray groundTruth =
                NDArrays.stack(groundTruthComponents)
                        .broadcast(3, batchSize, inW, inH, boxAttr - 1);

        return new NDList(boxLossScale, groundTruth);
    }

    /**
     * Calculates the IOU between priori Anchors and groundTruth.
     *
     * @param predx the tx value of prediction
     * @param predy the ty value of prediction
     * @param groundTruth the groundTruth value of labels
     * @param componentIndex the current component Index
     * @return an NDArray of IOU
     */
    public NDArray calculateIOU(
            NDArray predx, NDArray predy, NDArray groundTruth, int componentIndex) {
        int inW = (int) predx.getShape().get(2);
        int inH = (int) predx.getShape().get(3);
        int strideW = (int) inputShape.get(0) / inW;
        int strideH = (int) inputShape.get(1) / inH;

        NDList iouComponent = new NDList();
        // shape of predx, predy should all be (3,batchSize,inW,inH)
        // shape of groundTruth should be (3,batchSize,inW,inH,4)
        for (int i = 0; i < 3; i++) {
            NDArray curPredx = predx.get(i);
            NDArray curPredy = predy.get(i);
            float width = anchors[componentIndex * 6 + 2 * i] / strideW;
            float height = anchors[componentIndex * 6 + 2 * i + 1] / strideH;
            NDArray predLeft = curPredx.sub(width / 2);
            NDArray predRight = curPredx.add(width / 2);
            NDArray predTop = curPredy.sub(height / 2);
            NDArray predBottom = curPredy.add(height / 2);

            NDArray truth = groundTruth.get(i);

            NDArray trueLeft = truth.get("...,0").sub(truth.get("...,2").mul(inW).div(2));
            NDArray trueRight = truth.get("...,0").add(truth.get("...,2").mul(inW).div(2));
            NDArray trueTop = truth.get("...,1").sub(truth.get("...,3").mul(inH).div(2));
            NDArray trueBottom = truth.get("...,1").add(truth.get("...,3").mul(inH).div(2));

            NDArray left = NDArrays.maximum(predLeft, trueLeft);
            NDArray right = NDArrays.minimum(predRight, trueRight);
            NDArray top = NDArrays.maximum(predTop, trueTop);
            NDArray bottom = NDArrays.minimum(predBottom, trueBottom);

            NDArray inter = right.sub(left).mul(bottom.sub(top));
            NDArray union =
                    truth.get("...,2")
                            .mul(inW)
                            .mul(truth.get("...,3").mul(inH))
                            .add(width * height)
                            .sub(inter)
                            .add(EPSILON); // should not be divided by zero

            iouComponent.add(inter.div(union));
        }

        return NDArrays.stack(iouComponent);
    }

    /**
     * Creates a new builder to build a {@link YOLOv3Loss}.
     *
     * @return a new builder;
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link YOLOv3Loss} object. */
    public static class Builder {
        private String name = "YOLOv3Loss";
        private float[] anchorsArray = PRESETANCHORS;
        private int numClasses = 20;
        private Shape inputShape = new Shape(419, 419);
        private float ignoreThreshold = 0.5f;

        /**
         * Sets the loss name of YoloV3Loss.
         *
         * @param name the name of loss function
         * @return this {@code Builder}
         */
        public Builder setName(String name) {
            this.name = name;
            return this;
        }

        /**
         * Sets the preset anchors for YoloV3.
         *
         * @param anchorsArray the anchors in float array
         * @return this {@code Builder}
         */
        public Builder setAnchorsArray(float[] anchorsArray) {
            if (anchorsArray.length != PRESETANCHORS.length) {
                throw new IllegalArgumentException(
                        String.format(
                                "setAnchorsArray requires anchors of length %d, but was given"
                                        + " filters of length %d instead",
                                PRESETANCHORS.length, anchorsArray.length));
            }
            this.anchorsArray = anchorsArray;
            return this;
        }

        /**
         * Sets the number of total classes.
         *
         * @param numClasses the number of total classes
         * @return this {@code Builder}
         */
        public Builder setNumClasses(int numClasses) {
            this.numClasses = numClasses;
            return this;
        }

        /**
         * Sets the shape of the input picture.
         *
         * @param inputShape the shape of input picture.
         * @return this {@code Builder}
         */
        public Builder setInputShape(Shape inputShape) {
            this.inputShape = inputShape;
            return this;
        }

        /**
         * Sets the ignoreThreshold for iou to check if we think it detects a picture.
         *
         * @param ignoreThreshold the ignore threshold
         * @return this {@code Builder}
         */
        public Builder optIgnoreThreshold(float ignoreThreshold) {
            this.ignoreThreshold = ignoreThreshold;
            return this;
        }

        /**
         * Builds a {@link YOLOv3Loss} instance.
         *
         * @return a {@link YOLOv3Loss} instance.
         */
        public YOLOv3Loss build() {
            return new YOLOv3Loss(this);
        }
    }
}
