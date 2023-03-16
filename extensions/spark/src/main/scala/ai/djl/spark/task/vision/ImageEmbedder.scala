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
import ai.djl.modality.cv.translator.ImageFeatureExtractorFactory
import org.apache.spark.ml.image.ImageSchema
import org.apache.spark.ml.param.shared.HasOutputCol
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/**
 * ImageEmbedder performs image embedding on images.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
class ImageEmbedder(override val uid: String) extends BaseImagePredictor[Array[Byte]]
  with HasOutputCol {

  def this() = this(Identifiable.randomUID("ImageEmbedder"))

  /**
   * Sets the outputCol parameter.
   *
   * @param value the value of the parameter
   */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  setDefault(outputClass, classOf[Array[Byte]])
  setDefault(translatorFactory, new ImageFeatureExtractorFactory())

  /**
   * Performs image embedding on the provided dataset.
   *
   * @param dataset input dataset
   * @return output dataset
   */
  def embed(dataset: Dataset[_]): DataFrame = {
    transform(dataset)
  }

  /** @inheritdoc */
  override protected def transformRows(iter: Iterator[Row]): Iterator[Row] = {
    val predictor = model.newPredictor()
    iter.map(row => {
      val image = ImageFactory.getInstance().fromPixels(bgrToRgb(ImageSchema.getData(row)),
        ImageSchema.getWidth(row), ImageSchema.getHeight(row))
      Row.fromSeq(row.toSeq :+ predictor.predict(image))
    })
  }

  /** @inheritdoc */
  override def transformSchema(schema: StructType): StructType = {
    val outputSchema = StructType(schema.fields :+ StructField($(outputCol), ArrayType(ByteType)))
    outputSchema
  }
}
