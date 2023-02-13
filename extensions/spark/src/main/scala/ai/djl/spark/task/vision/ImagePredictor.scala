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
package ai.djl.spark.task.vision

import ai.djl.spark.task.BasePredictor
import org.apache.spark.ml.param.shared.HasInputCols
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Row

/**
 * ImagePredictor is the base class for image predictors.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
abstract class ImagePredictor[B](override val uid: String) extends BasePredictor[Row, B]
  with HasInputCols {

  def this() = this(Identifiable.randomUID("ImagePredictor"))

  /**
   * Sets the inputCols parameter.
   *
   * @param value the value of the parameter
   */
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  setDefault(inputClass, classOf[Row])
}
