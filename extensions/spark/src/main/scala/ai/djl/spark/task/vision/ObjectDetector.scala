/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.spark.task.vision

import ai.djl.modality.cv.ImageFactory
import ai.djl.modality.cv.output.DetectedObjects
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject
import ai.djl.modality.cv.translator.YoloV5TranslatorFactory
import org.apache.spark.ml.image.ImageSchema
import org.apache.spark.ml.param.shared.HasOutputCol
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{ArrayType, DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.jdk.CollectionConverters.{collectionAsScalaIterableConverter, seqAsJavaListConverter}

/**
 * ObjectDetector performs object detection on images.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
class ObjectDetector(override val uid: String) extends BaseImagePredictor[DetectedObjects]
  with HasOutputCol {

  def this() = this(Identifiable.randomUID("ObjectDetector"))

  /**
   * Sets the outputCol parameter.
   *
   * @param value the value of the parameter
   */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  setDefault(outputClass, classOf[DetectedObjects])
  setDefault(translatorFactory, new YoloV5TranslatorFactory())

  /**
   * Performs object detection on the provided dataset.
   *
   * @param dataset input dataset
   * @return output dataset
   */
  def detect(dataset: Dataset[_]): DataFrame = {
    transform(dataset)
  }

  /** @inheritdoc */
  override protected def transformRows(iter: Iterator[Row]): Iterator[Row] = {
    val predictor = model.newPredictor()
    iter.grouped($(batchSize)).flatMap { batch =>
      val inputs = batch.map(row =>
        ImageFactory.getInstance().fromPixels(bgrToRgb(ImageSchema.getData(row)),
          ImageSchema.getWidth(row), ImageSchema.getHeight(row))).asJava
      val output = predictor.batchPredict(inputs).asScala
      batch.zip(output).map { case (row, out) =>
        Row.fromSeq(row.toSeq :+ Row(out.getClassNames.toArray(), out.getProbabilities.toArray(),
          out.items[DetectedObject]().asScala.map(_.getBoundingBox.toString)))
      }
    }
  }

  /** @inheritdoc */
  override def transformSchema(schema: StructType): StructType = {
    val outputSchema = StructType(schema.fields :+
      StructField($(outputCol), StructType(Seq(StructField("class_names", ArrayType(StringType)),
        StructField("probabilities", ArrayType(DoubleType)),
        StructField("bounding_boxes", ArrayType(StringType))))))
    outputSchema
  }
}
