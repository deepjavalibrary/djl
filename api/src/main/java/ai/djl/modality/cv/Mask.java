/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv;

import java.awt.Graphics2D;

public class Mask extends Rectangle {

    private float[][] probDist;

    public Mask(double y, double x, double height, double width, float[][] dist) {
        super(y, x, height, width);
        this.probDist = dist;
    }

    public float[][] getProbDist() {
        return probDist;
    }

    @Override
    public void draw(Graphics2D g, int imageHeight, int imageWidth) {
        super.draw(g, imageHeight, imageWidth);
        // TODO: implement draw mask method
    }
}
