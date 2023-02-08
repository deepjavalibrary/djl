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
import ai.djl.spark.translator.vision.ImageClassificationTranslator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * ImageClassifier performs image classification on images.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
class ImageClassifier(override val uid: String) extends ImagePredictor[Classifications] {

  def this() = this(Identifiable.randomUID("ImageClassifier"))

  setDefault(outputClass, classOf[Classifications])
  setDefault(translator, new ImageClassificationTranslator())

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
  override def transformSchema(schema: StructType) = schema

  /** @inheritdoc */
  override def copy(paramMap: ParamMap) = this
}
