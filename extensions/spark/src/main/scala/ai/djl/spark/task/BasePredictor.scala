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
package ai.djl.spark.task

import ai.djl.spark.ModelLoader
import ai.djl.translate.Translator
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/**
 * BasePredictor is the base class of predictors.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
abstract class BasePredictor[A, B](override val uid: String) extends Transformer {

  def this() = this(Identifiable.randomUID("BasePredictor"))

  final val engine = new Param[String](this, "engine", "The engine")
  final val modelUrl = new Param[String](this, "modelUrl", "The model URL")
  final val inputClass = new Param[Class[A]](this, "inputClass", "The input class")
  final val outputClass = new Param[Class[B]](this, "outputClass", "The output class")
  final val translator = new Param[Translator[A, B]](this, "translator", "The translator")

  protected var model: ModelLoader[A, B] = _
  protected var outputSchema: StructType = _

  /**
   * Sets the engine parameter.
   *
   * @param value the value of the parameter
   */
  def setEngine(value: String): this.type = set(engine, value)

  setDefault(engine, null)

  /**
   * Sets the modelUrl parameter.
   *
   * @param value the value of the parameter
   */
  def setModelUrl(value: String): this.type = set(modelUrl, value)

  setDefault(modelUrl, null)

  /**
   * Sets the input class.
   *
   * @param value the value of the parameter
   */
  def setInputClass(value: Class[A]): this.type = set(inputClass, value)

  /**
   * Sets the output class.
   *
   * @param value the value of the parameter
   */
  def setOutputClass(value: Class[B]): this.type = set(outputClass, value)

  /**
   * Sets the translator parameter.
   *
   * @param value the value of the parameter
   */
  def setTranslator(value: Translator[A, B]): this.type = set(translator, value)

  /** @inheritdoc */
  override def transform(dataset: Dataset[_]): DataFrame = {
    model = new ModelLoader[A, B]($(engine), $(modelUrl), $(inputClass), $(outputClass))
    outputSchema = transformSchema(dataset.schema)
    val outputDf = dataset.toDF()
      .mapPartitions(transformRows)(RowEncoder.apply(outputSchema))
    outputDf
  }

  /** @inheritdoc */
  override def copy(extra: ParamMap): BasePredictor[A, B] = defaultCopy(extra)

  protected def transformRows(iter: Iterator[Row]): Iterator[Row]
}
