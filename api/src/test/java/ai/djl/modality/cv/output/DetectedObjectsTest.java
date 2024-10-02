/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv.output;

import ai.djl.modality.Classifications;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.List;

public class DetectedObjectsTest {

    private static final Logger logger = LoggerFactory.getLogger(DetectedObjectsTest.class);

    @Test
    public void test() {
        List<String> classes = Arrays.asList("cat", "dog");
        List<Double> probabilities = Arrays.asList(0.1d, 0.2d);
        List<BoundingBox> boxes =
                Arrays.asList(new Rectangle(1, 2, 3, 4), new Rectangle(3, 4, 5, 6));
        Classifications classifications = new Classifications(classes, probabilities);
        logger.info("Classifications: {}", classifications);

        Assert.assertEquals(
                classifications.toJson(),
                "[{\"className\":\"dog\",\"probability\":0.2},{\"className\":\"cat\",\"probability\":0.1}]");

        DetectedObjects detection = new DetectedObjects(classes, probabilities, boxes);

        logger.info("DetectedObjects: {}", detection);
        Assert.assertEquals(
                detection.toJson(),
                "[{\"boundingBox\":{\"rect\":[3,4,8,10]},\"className\":\"dog\",\"probability\":0.2},{\"boundingBox\":{\"rect\":[1,2,4,6]},\"className\":\"cat\",\"probability\":0.1}]");

        List<Point> points = Arrays.asList(new Point(1, 2), new Point(3, 4));
        boxes = Arrays.asList(new Landmark(1, 2, 3, 4, points), new Landmark(3, 4, 5, 6, points));
        detection = new DetectedObjects(classes, probabilities, boxes);

        logger.info("Landmarks: {}", detection);
        Assert.assertEquals(
                detection.toJson(),
                "[{\"boundingBox\":{\"rect\":[3,4,8,10],\"landmarks\":[{\"x\":1,\"y\":2},{\"x\":3,\"y\":4}]},\"className\":\"dog\",\"probability\":0.2},{\"boundingBox\":{\"rect\":[1,2,4,6],\"landmarks\":[{\"x\":1,\"y\":2},{\"x\":3,\"y\":4}]},\"className\":\"cat\",\"probability\":0.1}]");

        float[][] masks = {{1, 2, 3}, {4, 5, 6}};
        boxes = Arrays.asList(new Mask(1, 2, 3, 4, masks), new Mask(3, 4, 5, 6, masks, true));
        detection = new DetectedObjects(classes, probabilities, boxes);

        logger.info("Masks: {}", detection);
        Assert.assertEquals(
                detection.toJson(),
                "[{\"boundingBox\":{\"rect\":[3,4,8,10],\"fullImageMask\":true,\"mask\":[[1.0,2.0,3.0],[4.0,5.0,6.0]]},\"className\":\"dog\",\"probability\":0.2},{\"boundingBox\":{\"rect\":[1,2,4,6],\"mask\":[[1.0,2.0,3.0],[4.0,5.0,6.0]]},\"className\":\"cat\",\"probability\":0.1}]");
    }
}
