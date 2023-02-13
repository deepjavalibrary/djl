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
package ai.djl.spark.task.vision

import ai.djl.modality.Classifications
import ai.djl.modality.Classifications.Classification
import ai.djl.spark.translator.vision.ImageClassificationTranslator
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.shared.HasOutputCol
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.{ArrayType, DoubleType, MapType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.mutable

/**
 * ImageClassifier performs image classification on images.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
class ImageClassifier(override val uid: String) extends ImagePredictor[Classifications]
  with HasOutputCol {

  def this() = this(Identifiable.randomUID("ImageClassifier"))

  final val topK = new Param[Int](this, "topK", "The number of classes to return")

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

  setDefault(outputClass, classOf[Classifications])
  setDefault(translator, new ImageClassificationTranslator())
  setDefault(topK, 5)

  /**
   * Performs image classification on the provided dataset.
   *
   * @param dataset input dataset
   * @return output dataset
   */
  def classify(dataset: Dataset[_]): DataFrame = {
    transform(dataset)
  }

  /** @inheritdoc */
  override protected def transformRows(iter: Iterator[Row]): Iterator[Row] = {
    val predictor = model.newPredictor($(translator))
    iter.map(row => {
      val prediction = predictor.predict(row)
      val top = mutable.LinkedHashMap[String, Double]()
      val it: java.util.Iterator[Classification] = prediction.topK($(topK)).iterator()
      while (it.hasNext) {
        val t = it.next()
        top += (t.getClassName -> t.getProbability)
      }
      new GenericRowWithSchema(row.toSeq.toArray
        ++ Array[Any](Row(prediction.getClassNames.toArray, prediction.getProbabilities.toArray, top)),
        outputSchema)
    })
  }

  /** @inheritdoc */
  override def transformSchema(schema: StructType): StructType = {
    val outputSchema = StructType(schema.fields ++
      Array(StructField($(outputCol), StructType(Seq(StructField("class_names", ArrayType(StringType)),
        StructField("probabilities", ArrayType(DoubleType)),
        StructField("topK", MapType(StringType, DoubleType)))))))
    outputSchema
  }
}
