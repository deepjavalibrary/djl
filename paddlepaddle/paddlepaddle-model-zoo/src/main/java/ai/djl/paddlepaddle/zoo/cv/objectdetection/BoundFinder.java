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
package ai.djl.paddlepaddle.zoo.cv.objectdetection;

import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.Point;
import ai.djl.modality.cv.output.Rectangle;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.stream.Collectors;

/** Compute the bound of single colored region. */
public class BoundFinder {

    private final int[] deltaX = {0, 1, -1, 0};
    private final int[] deltaY = {1, 0, 0, -1};
    private List<List<Point>> pointsCollection;
    private int width;
    private int height;

    /**
     * Compute the bound based on the boolean mask.
     *
     * @param grid the 2D boolean mask that defines the region
     */
    public BoundFinder(boolean[][] grid) {
        pointsCollection = new ArrayList<>();
        width = grid.length;
        height = grid[0].length;
        boolean[][] visited = new boolean[width][height];
        // get all points connections
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                if (grid[i][j] && !visited[i][j]) {
                    pointsCollection.add(bfs(grid, i, j, visited));
                }
            }
        }
    }

    /**
     * Gets all points from the region.
     *
     * @return all connected points
     */
    public List<List<Point>> getPoints() {
        return pointsCollection;
    }

    /**
     * Compute rectangle bounding boxes.
     *
     * @return the region defined by boxes
     */
    public List<BoundingBox> getBoxes() {
        return pointsCollection
                .stream()
                .parallel()
                .map(
                        points -> {
                            double[] minMax = {Integer.MAX_VALUE, Integer.MAX_VALUE, -1, -1};
                            points.forEach(
                                    p -> {
                                        minMax[0] = Math.min(minMax[0], p.getX());
                                        minMax[1] = Math.min(minMax[1], p.getY());
                                        minMax[2] = Math.max(minMax[2], p.getX());
                                        minMax[3] = Math.max(minMax[3], p.getY());
                                    });
                            return new Rectangle(
                                    minMax[1],
                                    minMax[0],
                                    minMax[3] - minMax[1],
                                    minMax[2] - minMax[0]);
                        })
                .filter(rect -> rect.getWidth() * width > 5.0 && rect.getHeight() * height > 5.0)
                .collect(Collectors.toList());
    }

    private List<Point> bfs(boolean[][] grid, int x, int y, boolean[][] visited) {
        Queue<Point> queue = new ArrayDeque<>();
        queue.offer(new Point(x, y));
        visited[x][y] = true;

        List<Point> points = new ArrayList<>();
        while (!queue.isEmpty()) {
            Point point = queue.poll();
            points.add(new Point(point.getX() / width, point.getY() / height));
            for (int direction = 0; direction < 4; direction++) {
                int newX = (int) point.getX() + deltaX[direction];
                int newY = (int) point.getY() + deltaY[direction];
                if (!isVaild(grid, newX, newY, visited)) {
                    continue;
                }
                queue.offer(new Point(newX, newY));
                visited[newX][newY] = true;
            }
        }
        return points;
    }

    private boolean isVaild(boolean[][] grid, int x, int y, boolean[][] visited) {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return false;
        }
        if (visited[x][y]) {
            return false;
        }
        return grid[x][y];
    }
}
