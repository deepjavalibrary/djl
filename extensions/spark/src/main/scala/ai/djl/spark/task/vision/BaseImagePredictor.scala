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

import ai.djl.modality.cv.Image
import ai.djl.spark.task.BasePredictor
import org.apache.spark.ml.param.shared.HasInputCols
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{BinaryType, IntegerType, StringType, StructType}

/**
 * BaseImagePredictor is the base class for image predictors.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
abstract class BaseImagePredictor[B](override val uid: String) extends BasePredictor[Image, B]
  with HasInputCols {

  def this() = this(Identifiable.randomUID("BaseImagePredictor"))

  /**
   * Sets the inputCols parameter.
   *
   * @param value the value of the parameter
   */
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  setDefault(batchSize, 10)
  setDefault(inputClass, classOf[Image])

  /** @inheritdoc */
  override protected def validateInputType(schema: StructType): Unit = {
    assert($(inputCols).length == 6, "inputCols must have 6 columns")
    validateType(schema($(inputCols)(0)), StringType)
    validateType(schema($(inputCols)(1)), IntegerType)
    validateType(schema($(inputCols)(2)), IntegerType)
    validateType(schema($(inputCols)(3)), IntegerType)
    validateType(schema($(inputCols)(4)), IntegerType)
    validateType(schema($(inputCols)(5)), BinaryType)
  }

  /**
   * Convert BGR byte array to RGB int array that represents pixels.
   *
   * @param bytes BGR byte array to convert
   * @return RGB int array that represents pixels
   */
  def bgrToRgb(bytes: Array[Byte]): Array[Int] = {
    val res = new Array[Int](bytes.length / 3)
    for (i <- 0 until bytes.length / 3) {
      val b = bytes(i * 3) & 0xFF
      val g = bytes(i * 3 + 1) & 0xFF
      val r = bytes(i * 3 + 2) & 0xFF
      res(i) = (r << 16) | (g << 8) | b
    }
    res
  }
}
