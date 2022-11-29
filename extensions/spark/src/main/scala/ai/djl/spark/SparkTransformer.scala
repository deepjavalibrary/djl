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
package ai.djl.spark

import ai.djl.translate.Translator
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoders, Row}

/**
 * SparkPredictor is a transformer that transforms one dataset into another
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
class SparkTransformer[T](override val uid: String) extends Transformer {

  def this() = this(Identifiable.randomUID("SparkTransformer"))

  final val inputCol = new Param[String](this, "inputCol", "The input column")
  final val outputCol = new Param[String](this, "outputCol", "The output column")
  final val modelUrl = new Param[String](this, "modelUrl", "The model URL")
  final val outputClass = new Param[Class[T]](this, "outputClass", "The output class")
  final val translator = new Param[Translator[Row, T]](this, "translator", "The translator")

  /**
   * Sets the inputCol parameter.
   *
   * @param value the value of the parameter
   */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /**
   * Sets the outputCol parameter.
   *
   * @param value the value of the parameter
   */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /**
   * Sets the modelUrl parameter.
   *
   * @param value the value of the parameter
   */
  def setModelUrl(value: String): this.type = set(modelUrl, value)

  /**
   * Sets the output class.
   *
   * @param value the value of the parameter
   */
  def setOutputClass(value: Class[T]): this.type = set(outputClass, value)

  /**
   * Sets the translator parameter.
   *
   * @param value the value of the parameter
   */
  def setTranslator(value: Translator[Row, T]): this.type = set(translator, value)

  /** @inheritdoc */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val model = new SparkModel[T]($(modelUrl), $(outputClass))
    val outputDf = dataset.select($(inputCol)).mapPartitions(partition => {
      val predictor = model.newPredictor($(translator))
      partition.map(row => {
        predictor.predict(row).toString
      })
    })(Encoders.STRING)
    outputDf.select($(outputCol))
  }

  /** @inheritdoc */
  override def transformSchema(schema: StructType) = schema

  /** @inheritdoc */
  override def copy(paramMap: ParamMap) = this
}
