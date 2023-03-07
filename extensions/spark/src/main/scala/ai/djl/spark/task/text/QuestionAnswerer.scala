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
package ai.djl.spark.task.text

import ai.djl.huggingface.translator.QuestionAnsweringTranslatorFactory
import ai.djl.modality.nlp.qa.QAInput
import org.apache.spark.ml.param.shared.{HasInputCols, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/**
 * QuestionAnswerer performs question answering on text.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
class QuestionAnswerer(override val uid: String) extends BaseTextPredictor[QAInput, String]
  with HasInputCols with HasOutputCol {

  def this() = this(Identifiable.randomUID("QuestionAnswerer"))

  private val inputColIndices = new Array[Int](2)

  /**
   * Sets the inputCols parameter. The number of columns has to be 2. The first column is the question
   * and the second column is the context.
   *
   * @param value the value of the parameter
   */
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  /**
   * Sets the outputCol parameter.
   *
   * @param value the value of the parameter
   */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  setDefault(inputClass, classOf[QAInput])
  setDefault(outputClass, classOf[String])
  setDefault(translatorFactory, new QuestionAnsweringTranslatorFactory())

  /**
   * Performs text embedding on the provided dataset.
   *
   * @param dataset input dataset
   * @return output dataset
   */
  def answer(dataset: Dataset[_]): DataFrame = {
    transform(dataset)
  }

  /** @inheritdoc */
  override def transform(dataset: Dataset[_]): DataFrame = {
    inputColIndices(0) = dataset.schema.fieldIndex($(inputCols)(0))
    inputColIndices(1) = dataset.schema.fieldIndex($(inputCols)(1))
    super.transform(dataset)
  }

  /** @inheritdoc */
  override protected def transformRows(iter: Iterator[Row]): Iterator[Row] = {
    val predictor = model.newPredictor()
    iter.map(row => {
      Row.fromSeq(row.toSeq :+ predictor.predict(new QAInput(row.getString(inputColIndices(0)),
        row.getString(inputColIndices(1)))))
    })
  }

  /** @inheritdoc */
  override def transformSchema(schema: StructType): StructType = {
    assert($(inputCols).length == 2, "inputCols must have 2 columns")
    validateInputType(schema($(inputCols)(0)))
    validateInputType(schema($(inputCols)(1)))
    val outputSchema = StructType(schema.fields :+ StructField($(outputCol), StringType))
    outputSchema
  }
}
