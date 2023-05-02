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

import ai.djl.huggingface.translator.TextClassificationTranslatorFactory
import ai.djl.modality.Classifications
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{ArrayType, DoubleType, MapType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.convert.ImplicitConversions.`collection AsScalaIterable`

/**
 * TextClassifier performs text classification on text.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
class TextClassifier(override val uid: String) extends BaseTextPredictor[Array[String], Array[Classifications]]
  with HasInputCol with HasOutputCol {

  def this() = this(Identifiable.randomUID("TextClassifier"))

  final val topK = new Param[Int](this, "topK", "The number of classes to return")

  private var inputColIndex : Int = _

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
   * Sets the topK parameter.
   *
   * @param value the value of the parameter
   */
  def setTopK(value: Int): this.type = set(topK, value)

  setDefault(inputClass, classOf[Array[String]])
  setDefault(outputClass, classOf[Array[Classifications]])
  setDefault(translatorFactory, new TextClassificationTranslatorFactory())

  /**
   * Performs text classification on the provided dataset.
   *
   * @param dataset input dataset
   * @return output dataset
   */
  def classify(dataset: Dataset[_]): DataFrame = {
    transform(dataset)
  }

  /** @inheritdoc */
  override def transform(dataset: Dataset[_]): DataFrame = {
    if (isDefined(topK)) {
      arguments.put("topK", $(topK).toString)
    }
    inputColIndex = dataset.schema.fieldIndex($(inputCol))
    super.transform(dataset)
  }

  /** @inheritdoc */
  override protected def transformRows(iter: Iterator[Row]): Iterator[Row] = {
    val predictor = model.newPredictor()
    iter.grouped($(batchSize)).flatMap { batch =>
      val inputs = batch.map(_.getString(inputColIndex)).toArray
      val output = predictor.predict(inputs)
      batch.zip(output).map { case (row, out) =>
        Row.fromSeq(row.toSeq :+ Row(out.getClassNames.toArray(), out.getProbabilities.toArray(),
          out.topK[Classifications.Classification]().map(_.toString)))
      }
    }
  }

  /** @inheritdoc */
  override protected def validateInputType(schema: StructType): Unit = {
    validateType(schema($(inputCol)), StringType)
  }

  /** @inheritdoc */
  override def transformSchema(schema: StructType): StructType = {
    val outputSchema = StructType(schema.fields :+
      StructField($(outputCol), StructType(Seq(StructField("class_names", ArrayType(StringType)),
        StructField("probabilities", ArrayType(DoubleType)),
        StructField("top_k", ArrayType(StringType))))))
    outputSchema
  }
}
