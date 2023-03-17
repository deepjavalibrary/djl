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
package ai.djl.spark.task.binary

import ai.djl.spark.task.BasePredictor
import ai.djl.spark.translator.binary.NpBinaryTranslatorFactory
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{BinaryType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/**
 * BinaryPredictor performs prediction on binary input.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
class BinaryPredictor(override val uid: String) extends BasePredictor[Array[Byte], Array[Byte]]
  with HasInputCol with HasOutputCol {

  def this() = this(Identifiable.randomUID("BinaryPredictor"))

  private var inputColIndex: Int = _

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

  setDefault(inputClass, classOf[Array[Byte]])
  setDefault(outputClass, classOf[Array[Byte]])
  setDefault(translatorFactory, new NpBinaryTranslatorFactory())

  /**
   * Performs prediction on the provided dataset.
   *
   * @param dataset input dataset
   * @return output dataset
   */
  def predict(dataset: Dataset[_]): DataFrame = {
    transform(dataset)
  }

  /** @inheritdoc */
  override def transform(dataset: Dataset[_]): DataFrame = {
    arguments.put("batchifier", $(batchifier))
    inputColIndex = dataset.schema.fieldIndex($(inputCol))
    super.transform(dataset)
  }

  /** @inheritdoc */
  override protected def transformRows(iter: Iterator[Row]): Iterator[Row] = {
    val predictor = model.newPredictor()
    iter.map(row => {
      Row.fromSeq(row.toSeq :+ predictor.predict(row.getAs[Array[Byte]](inputColIndex)))
    })
  }

  /** @inheritdoc */
  def validateInputType(schema: StructType): Unit = {
    validateType(schema($(inputCol)), BinaryType)
  }

  /** @inheritdoc */
  override def transformSchema(schema: StructType): StructType = {
    val outputSchema = StructType(schema.fields :+ StructField($(outputCol), BinaryType))
    outputSchema
  }
}
