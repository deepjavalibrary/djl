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

import ai.djl.spark.ModelLoader
import ai.djl.spark.translator.text.TextEmbeddingTranslator
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, Encoders, Row}

import java.util

/**
 * TextEmbedder performs text embedding on text.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
class TextEmbedder(override val uid: String) extends TextPredictor[Array[Float]] {

  def this() = this(Identifiable.randomUID("TextEmbedder"))

  setDefault(outputClass, classOf[Array[Float]])
  setDefault(translator, new TextEmbeddingTranslator())

  /** @inheritdoc */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val model = new ModelLoader[Row, Array[Float]]($(engine), $(modelUrl), $(inputClass), $(outputClass))
    val outputDf = dataset.selectExpr($(inputCols):_*).mapPartitions(partition => {
      val predictor = model.newPredictor($(translator))
      partition.map(row => {
        util.Arrays.toString(predictor.predict(row))
      })
    })(Encoders.STRING)
    outputDf.selectExpr($(outputCols):_*)
  }

  /**
   * Performs text embedding on the provided dataset.
   *
   * @param dataset input dataset
   * @return output dataset
   */
  def embed(dataset: Dataset[_]): DataFrame = {
    transform(dataset)
  }
}