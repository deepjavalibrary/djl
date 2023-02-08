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

import ai.djl.spark.ModelLoader
import ai.djl.spark.task.BasePredictor
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoders, Row}

/**
 * ImagePredictor performs prediction on images.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
class ImagePredictor[B](override val uid: String) extends BasePredictor[Row, B] {

  def this() = this(Identifiable.randomUID("ImagePredictor"))

  setDefault(inputClass, classOf[Row])

  /** @inheritdoc */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val model = new ModelLoader[Row, B]($(engine), $(modelUrl), $(inputClass), $(outputClass))
    val outputDf = dataset.selectExpr($(inputCols):_*).mapPartitions(partition => {
      val predictor = model.newPredictor($(translator))
      partition.map(row => {
        predictor.predict(row).toString
      })
    })(Encoders.STRING)
    outputDf.selectExpr($(outputCols):_*)
  }

  /** @inheritdoc */
  override def transformSchema(schema: StructType) = schema

  /** @inheritdoc */
  override def copy(paramMap: ParamMap) = this
}
