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
package ai.djl.examples.inference.face;

import ai.djl.examples.inference.face.model.FaceDetectedObjects;
import ai.djl.examples.inference.face.model.FaceObject;
import ai.djl.examples.inference.face.model.Landmark;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.Point;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public class FaceDetectionTranslator implements Translator<Image, FaceDetectedObjects> {
  protected Batchifier batchifier = Batchifier.STACK;
  private double confThresh;
  private double visThresh;
  private double nmsThresh;
  private int topK;
  private double[] variance;
  private int[][] scales;
  private int[] steps;
  private int width;
  private int height;

  public FaceDetectionTranslator(
      double confThresh,
      double visThresh,
      double nmsThresh,
      double[] variance,
      int topK,
      int[][] scales,
      int[] steps) {
    this.confThresh = confThresh;
    this.visThresh = visThresh;
    this.nmsThresh = nmsThresh;
    this.variance = variance;
    this.topK = topK;
    this.scales = scales;
    this.steps = steps;
  }

  @Override
  public NDList processInput(TranslatorContext ctx, Image input) {
    width = input.getWidth();
    height = input.getHeight();
    NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
    array = array.transpose(2, 0, 1).flip(0); // HWC -> CHW RGB -> BGR
    // The network by default takes float32
    if (!array.getDataType().equals(DataType.FLOAT32)) {
      array = array.toType(DataType.FLOAT32, false);
    }
    NDArray mean = ctx.getNDManager().create(new float[] {104f, 117f, 123f}, new Shape(3, 1, 1));
    array = array.sub(mean);
    NDList list = new NDList(array);

    return list;
  }

  @Override
  public FaceDetectedObjects processOutput(TranslatorContext ctx, NDList list) {
    double[][] priors = this.boxRecover(width, height, scales, steps);
    NDArray loc = list.get(0);
    float[] locFloat = loc.toFloatArray();
    double[][] boxes = this.decodeBoxes(locFloat, priors, variance);
    NDArray conf = list.get(1);
    float[] scores = this.decodeConf(conf);
    NDArray landms = list.get(2);
    List<List<Point>> landmsList =
        this.decodeLandm(landms.toFloatArray(), priors, variance, width, height);

    PriorityQueue<FaceObject> pq =
        new PriorityQueue<FaceObject>(
            10,
            new Comparator<FaceObject>() {
              @Override
              public int compare(final FaceObject lhs, final FaceObject rhs) {
                return Double.compare(rhs.getScore(), lhs.getScore());
              }
            });

    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > this.confThresh) {
        BoundingBox rect =
            new Rectangle(
                boxes[i][0], boxes[i][1], boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]);

        FaceObject faceObject = new FaceObject(scores[i], rect, landmsList.get(i));
        pq.add(faceObject);
      }
    }

    ArrayList<FaceObject> topKArrayList = new ArrayList<FaceObject>();
    int index = 0;
    while (pq.size() > 0) {
      FaceObject faceObject = pq.poll();
      if (index >= this.topK) {
        break;
      }
      topKArrayList.add(faceObject);
    }

    ArrayList<FaceObject> nmsList = this.nms(topKArrayList, this.nmsThresh);

    List<String> classNames = new ArrayList<String>();
    List<Double> probabilities = new ArrayList<Double>();
    List<BoundingBox> boundingBoxes = new ArrayList<BoundingBox>();
    List<Landmark> landmarks = new ArrayList<Landmark>();

    for (int i = 0; i < nmsList.size(); i++) {
      FaceObject faceObject = nmsList.get(i);

      if (faceObject.getScore() < this.visThresh) {
        continue;
      }

      classNames.add(new String("Face"));
      probabilities.add((double) faceObject.getScore());
      boundingBoxes.add(faceObject.getBoundingBox());
      List<Point> keyPoints = faceObject.getKeyPoints();
      Landmark landmark = new Landmark(keyPoints);
      landmarks.add(landmark);
    }

    return new FaceDetectedObjects(classNames, probabilities, boundingBoxes, landmarks);
  }

  private double[][] boxRecover(int width, int height, int[][] scales, int[] steps) {
    int[][] aspectRatio = new int[steps.length][2];
    for (int i = 0; i < steps.length; i++) {
      int wRatio = (int) Math.ceil((float) width / steps[i]);
      int hRatio = (int) Math.ceil((float) height / steps[i]);
      aspectRatio[i] = new int[] {hRatio, wRatio};
    }

    List<double[]> defaultBoxes = new ArrayList<>();

    for (int idx = 0; idx < steps.length; idx++) {
      int[] scale = scales[idx];
      for (int h = 0; h < aspectRatio[idx][0]; h++) {
        for (int w = 0; w < aspectRatio[idx][1]; w++) {
          for (int index = 0; index < scale.length; index++) {
            double skx = scale[index] * 1.0 / width;
            double sky = scale[index] * 1.0 / height;
            double cx = (w + 0.5) * steps[idx] / width;
            double cy = (h + 0.5) * steps[idx] / height;
            defaultBoxes.add(new double[] {cx, cy, skx, sky});
          }
        }
      }
    }

    double[][] boxes = new double[defaultBoxes.size()][defaultBoxes.get(0).length];
    for (int i = 0; i < defaultBoxes.size(); i++) {
      boxes[i] = defaultBoxes.get(i);
    }
    return boxes;
  }
  // decode prior boxes
  private double[][] decodeBoxes(float[] locs, double[][] priors, double[] variance) {
    double[][] boxes = new double[priors.length][4];
    for (int i = 0; i < priors.length; i++) {
      double x = priors[i][0] + locs[i * 4 + 0] * variance[0] * priors[i][2];
      double y = priors[i][1] + locs[i * 4 + 1] * variance[0] * priors[i][3];
      double w = priors[i][2] * (float) Math.exp(locs[i * 4 + 2] * variance[1]);
      double h = priors[i][3] * (float) Math.exp(locs[i * 4 + 3] * variance[1]);
      x = x - w / 2;
      y = y - h / 2;
      w = w + x;
      h = h + y;
      boxes[i] = new double[] {x, y, w, h};
    }

    return boxes;
  }

  // decode face confidence
  private float[] decodeConf(NDArray conf) {
    return conf.get(new NDIndex(":, 1")).toFloatArray();
  }

  // decode face landmarks, 5 points per face
  private List<List<Point>> decodeLandm(
      float[] landms, double[][] priors, double[] variance, int width, int height) {
    List<List<Point>> landmsArr = new ArrayList<>();
    List<Point> points = new ArrayList<>();
    for (int i = 0; i < priors.length; i++) {
      points = new ArrayList<>();
      for (int j = 0; j < 5; j++) { // 5 face landmarks
        double x = priors[i][0] + landms[i * 5 * 2 + j * 2] * variance[0] * priors[i][2];
        double y = priors[i][1] + landms[i * 5 * 2 + j * 2 + 1] * variance[0] * priors[i][3];
        points.add(new Point(x * width, y * height));
      }
      landmsArr.add(points);
    }
    return landmsArr;
  }

  // NMS - non maximum suppression
  public static ArrayList<FaceObject> nms(ArrayList<FaceObject> list, double nmsThresh) {
    ArrayList<FaceObject> nmsList = new ArrayList<FaceObject>();
    // Find max confidence
    PriorityQueue<FaceObject> pq =
        new PriorityQueue<FaceObject>(
            10,
            new Comparator<FaceObject>() {
              @Override
              public int compare(final FaceObject lhs, final FaceObject rhs) {
                // Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(rhs.getScore(), lhs.getScore());
              }
            });

    for (int i = 0; i < list.size(); i++) {
      // put high confidence at the head of the queue.
      pq.add(list.get(i));
    }

    while (pq.size() > 0) {
      // insert face object with max confidence
      FaceObject[] a = new FaceObject[pq.size()];
      FaceObject[] detections = pq.toArray(a);
      FaceObject max = detections[0];
      nmsList.add(max);

      // clear pq to do next nms
      pq.clear();

      for (int j = 1; j < detections.length; j++) {
        FaceObject detection = detections[j];
        BoundingBox b = detection.getBoundingBox();
        double boxIoU = max.getBoundingBox().getIoU(b);
        if (boxIoU <= nmsThresh) {
          pq.add(detection);
        }
      }
    }
    return nmsList;
  }

  @Override
  public Batchifier getBatchifier() {
    return batchifier;
  }
}
