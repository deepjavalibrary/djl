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

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/**
 * TextTokenizer performs text tokenization using HuggingFace tokenizers in Spark.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
class TextTokenizer(override val uid: String) extends BaseTextPredictor[String, Array[String]]
  with HasInputCol with HasOutputCol {

  def this() = this(Identifiable.randomUID("TextTokenizer"))

  final val hfModelId = new Param[String](this, "hfModelId", "The Huggingface model ID")

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

  /**
   * Sets the hfModelId parameter.
   *
   * @param value the value of the parameter
   */
  def setHfModelId(value: String): this.type = set(hfModelId, value)

  setDefault(inputClass, classOf[String])
  setDefault(outputClass, classOf[Array[String]])
  setDefault(translatorFactory, null)

  /**
   * Performs sentence tokenization on the provided dataset.
   *
   * @param dataset input dataset
   * @return output dataset
   */
  def tokenize(dataset: Dataset[_]): DataFrame = {
    transform(dataset)
  }

  /** @inheritdoc */
  override def transform(dataset: Dataset[_]): DataFrame = {
    inputColIndex = dataset.schema.fieldIndex($(inputCol))
    super.transform(dataset)
  }

  /** @inheritdoc */
  override def transformRows(iter: Iterator[Row]): Iterator[Row] = {
    val tokenizer = HuggingFaceTokenizer.newInstance($(hfModelId))
    iter.map(row => {
      Row.fromSeq(row.toSeq :+ tokenizer.tokenize(row.getString(inputColIndex)).toArray)
    })
  }

  /** @inheritdoc */
  def validateInputType(schema: StructType): Unit = {
    validateType(schema($(inputCol)), StringType)
  }

  /** @inheritdoc */
  override def transformSchema(schema: StructType): StructType = {
    val outputSchema = StructType(schema.fields :+
      StructField($(outputCol), ArrayType(StringType)))
    outputSchema
  }
}
