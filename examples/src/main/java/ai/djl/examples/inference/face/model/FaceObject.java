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
package ai.djl.examples.inference.face.model;

import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.Point;

import java.util.List;

public class FaceObject {
  private float score;
  private BoundingBox boundingBox;
  private List<Point> keyPoints;
  private float[] embeddings;

  public FaceObject() {}

  public FaceObject(float score, BoundingBox box, List<Point> keyPoints) {
    this.score = score;
    this.boundingBox = box;
    this.keyPoints = keyPoints;
  }

  public float getScore() {
    return score;
  }

  public BoundingBox getBoundingBox() {
    return boundingBox;
  }

  public void setBoundingBox(BoundingBox boundingBox) {
    this.boundingBox = boundingBox;
  }

  public List<Point> getKeyPoints() {
    return keyPoints;
  }

  public float[] getEmbeddings() {
    return embeddings;
  }

  public void setEmbeddings(float[] embeddings) {
    this.embeddings = embeddings;
  }
}
