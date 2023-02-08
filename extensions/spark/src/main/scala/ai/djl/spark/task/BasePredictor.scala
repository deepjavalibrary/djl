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

import ai.djl.translate.Translator
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType

/**
 * BasePredictor is the base class of predictors.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
abstract class BasePredictor[A, B](override val uid: String) extends Transformer {

  def this() = this(Identifiable.randomUID("BasePredictor"))

  final val inputCols = new Param[Array[String]](this, "inputCols", "The input columns")
  final val outputCols = new Param[Array[String]](this, "outputCols", "The output columns")
  final val engine = new Param[String](this, "engine", "The engine")
  final val modelUrl = new Param[String](this, "modelUrl", "The model URL")
  final val inputClass = new Param[Class[A]](this, "inputClass", "The input class")
  final val outputClass = new Param[Class[B]](this, "outputClass", "The output class")
  final val translator = new Param[Translator[A, B]](this, "translator", "The translator")

  /**
   * Sets the inputCols parameter.
   *
   * @param value the value of the parameter
   */
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  setDefault(inputCols, Array("*"))

  /**
   * Sets the outputCols parameter.
   *
   * @param value the value of the parameter
   */
  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

  setDefault(inputCols, Array("value"))

  /**
   * Sets the engine parameter.
   *
   * @param value the value of the parameter
   */
  def setEngine(value: String): this.type = set(engine, value)

  /**
   * Sets the modelUrl parameter.
   *
   * @param value the value of the parameter
   */
  def setModelUrl(value: String): this.type = set(modelUrl, value)

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
  override def transformSchema(schema: StructType) = schema

  /** @inheritdoc */
  override def copy(paramMap: ParamMap) = this
}
