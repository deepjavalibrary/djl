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
package ai.djl.modality.cv;

import java.util.ArrayList;
import java.util.List;

public class MultiBoxPriors {
    private List<Float> sizes;
    private List<Float> ratios;
    private int inputSize;
    private int mapWidth;
    private int mapHeight;

    public MultiBoxPriors(
            List<Float> sizes, List<Float> ratios, int inputSize, int mapWidth, int mapHeight) {
        this.sizes = sizes;
        this.ratios = ratios;
        this.inputSize = inputSize;
        this.mapWidth = mapWidth;
        this.mapHeight = mapHeight;
    }

    public int numberOfAnchorBoxes() {
        return sizes.size() + ratios.size() - 1;
    }

    public List<Rectangle> generateAnchorBoxes() {
        List<Rectangle> anchorBoxes = new ArrayList<>();
        for (int i = 0; i < mapWidth; i++) {
            for (int j = 0; j < mapHeight; j++) {
                anchorBoxes.addAll(multiBoxPriors(new Point(i, j)));
            }
        }
        return anchorBoxes;
    }

    public List<Rectangle> multiBoxPriors(Point point) {
        List<Rectangle> multiBoxPriors = new ArrayList<>();
        for (Float ratio : ratios) {
            double width = inputSize * sizes.get(0) * Math.sqrt(ratio);
            double height = inputSize * sizes.get(0) / Math.sqrt(ratio);
            double x = point.getX() * inputSize / mapWidth - width / 2;
            double y = point.getY() * inputSize / mapHeight + height / 2;
            Rectangle box = new Rectangle(x, y, width, height);
            multiBoxPriors.add(box);
        }
        for (int i = 1; i < sizes.size(); i++) {
            double width = inputSize * sizes.get(i) * Math.sqrt(ratios.get(0));
            double height = inputSize * sizes.get(i) / Math.sqrt(ratios.get(0));
            double x = point.getX() * inputSize / mapWidth - width / 2;
            double y = point.getY() * inputSize / mapHeight + height / 2;
            Rectangle box = new Rectangle(x, y, width, height);
            multiBoxPriors.add(box);
        }
        return multiBoxPriors;
    }
}
