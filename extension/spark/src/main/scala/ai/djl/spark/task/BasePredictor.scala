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
import ai.djl.translate.TranslatorFactory
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{IntParam, Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types.{DataType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/**
 * BasePredictor is the base class of predictors.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
abstract class BasePredictor[A, B](override val uid: String) extends Transformer {

  def this() = this(Identifiable.randomUID("BasePredictor"))

  final val modelUrl = new Param[String](this, "modelUrl", "The model URL")
  final val engine = new Param[String](this, "engine", "The engine")
  final val batchSize = new IntParam(this, "batchSize", "The batch size")
  final val inputClass = new Param[Class[A]](this, "inputClass", "The input class")
  final val outputClass = new Param[Class[B]](this, "outputClass", "The output class")
  final val translatorFactory = new Param[TranslatorFactory](this, "translatorFactory", "The translator factory")
  final val batchifier = new Param[String](this, "batchifier",
    "The batchifier. Valid values include none (default), stack, and padding.")

  protected var model: ModelLoader[A, B] = _
  protected var arguments: java.util.Map[String, AnyRef] = new java.util.HashMap[String, AnyRef]
  protected var outputSchema: StructType = _

  /**
   * Sets the modelUrl parameter.
   *
   * @param value the value of the parameter
   */
  def setModelUrl(value: String): this.type = set(modelUrl, value)

  /**
   * Sets the engine parameter.
   *
   * @param value the value of the parameter
   */
  def setEngine(value: String): this.type = set(engine, value)

  /**
   * Sets the batchSize parameter.
   *
   * @param value the value of the parameter
   */
  def setBatchSize(value: Int): this.type = set(batchSize, value)

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
   * Sets the translatorFactory parameter.
   *
   * @param value the value of the parameter
   */
  def setTranslatorFactory(value: TranslatorFactory): this.type = set(translatorFactory, value)

  /**
   * Sets the batchifier parameter.
   *
   * @param value the value of the parameter
   */
  def setBatchifier(value: String): this.type = set(batchifier, value)

  setDefault(modelUrl, null)
  setDefault(engine, null)
  setDefault(batchSize, 10)

  /** @inheritdoc */
  override def transform(dataset: Dataset[_]): DataFrame = {
    if (isDefined(batchifier)) {
      arguments.put("batchifier", $(batchifier))
    }
    model = new ModelLoader[A, B]($(engine), $(modelUrl), $(inputClass),
      $(outputClass), $(translatorFactory), arguments)
    validateInputType(dataset.schema)
    outputSchema = transformSchema(dataset.schema)
    val outputDf = dataset.toDF()
      .mapPartitions(transformRows)(RowEncoder.apply(outputSchema))
    outputDf
  }

  /** @inheritdoc */
  override def copy(extra: ParamMap): BasePredictor[A, B] = defaultCopy(extra)

  /**
   * Transforms the rows.
   *
   * @param iter the rows to transform
   * @return the transformed rows
   */
  protected def transformRows(iter: Iterator[Row]): Iterator[Row]

  /**
   * Validate input type.
   *
   * @param schema the schema to validate
   */
  protected def validateInputType(schema: StructType): Unit

  /**
   * Validate data type.
   *
   * @param field the field to validate
   * @param tp the expected type
   */
  def validateType(field: StructField, tp: DataType): Unit = {
    require(field.dataType == tp,
      s"Input column ${field.name} type must be ${tp} but got ${field.dataType}.")
  }
}
