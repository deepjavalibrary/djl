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
package ai.djl.spark.task.vision

import ai.djl.modality.Classifications
import ai.djl.modality.cv.ImageFactory
import ai.djl.modality.cv.translator.ImageClassificationTranslatorFactory
import org.apache.spark.ml.image.ImageSchema
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.shared.HasOutputCol
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{ArrayType, DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.convert.ImplicitConversions.`collection AsScalaIterable`

/**
 * ImageClassifier performs image classification on images.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
class ImageClassifier(override val uid: String) extends BaseImagePredictor[Classifications]
  with HasOutputCol {

  def this() = this(Identifiable.randomUID("ImageClassifier"))

  final val applySoftmax = new Param[Boolean](this, "applySoftmax", "Whether to apply softmax when processing output")
  final val topK = new Param[Int](this, "topK", "The number of classes to return")

  /**
   * Sets the outputCol parameter.
   *
   * @param value the value of the parameter
   */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /**
   * Sets whether to apply softmax when processing output. Some models already include softmax
   * in the last layer, so don't apply softmax when processing model output.
   *
   * @param value the value of the parameter
   */
  def setApplySoftmax(value: Boolean): this.type = set(applySoftmax, value)

  /**
   * Sets the topK parameter.
   *
   * @param value the value of the parameter
   */
  def setTopK(value: Int): this.type = set(topK, value)

  setDefault(outputClass, classOf[Classifications])
  setDefault(translatorFactory, new ImageClassificationTranslatorFactory())
  setDefault(applySoftmax, true)
  setDefault(topK, 5)

  /**
   * Performs image classification on the provided dataset.
   *
   * @param dataset input dataset
   * @return output dataset
   */
  def classify(dataset: Dataset[_]): DataFrame = {
    transform(dataset)
  }

  /** @inheritdoc */
  override def transform(dataset: Dataset[_]): DataFrame = {
    arguments.put("applySoftmax", $(applySoftmax).toString)
    arguments.put("topK", $(topK).toString)
    super.transform(dataset)
  }

  /** @inheritdoc */
  override protected def transformRows(iter: Iterator[Row]): Iterator[Row] = {
    val predictor = model.newPredictor()
    iter.map(row => {
      val image = ImageFactory.getInstance().fromPixels(bgrToRgb(ImageSchema.getData(row)),
        ImageSchema.getWidth(row), ImageSchema.getHeight(row))
      val prediction = predictor.predict(image)
      val top = prediction.topK[Classifications.Classification]($(topK)).map(item => item.toString)
      Row.fromSeq(row.toSeq :+ Row(prediction.getClassNames.toArray,
        prediction.getProbabilities.toArray, top))
    })
  }

  /** @inheritdoc */
  override def transformSchema(schema: StructType): StructType = {
    val outputSchema = StructType(schema.fields :+
      StructField($(outputCol), StructType(Seq(StructField("class_names", ArrayType(StringType)),
        StructField("probabilities", ArrayType(DoubleType)),
        StructField("topK", ArrayType(StringType))))))
    outputSchema
  }
}
