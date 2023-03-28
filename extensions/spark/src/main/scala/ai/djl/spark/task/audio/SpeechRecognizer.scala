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
package ai.djl.spark.task.audio

import ai.djl.modality.audio.AudioFactory
import ai.djl.modality.audio.translator.SpeechRecognitionTranslatorFactory
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.{BinaryType, StringType, StructField, StructType}

import java.io.ByteArrayInputStream

/**
 * SpeechRecognizer performs speech recognition on audio.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
class SpeechRecognizer(override val uid: String) extends BaseAudioPredictor[String]
  with HasInputCol with HasOutputCol {

  def this() = this(Identifiable.randomUID("SpeechRecognizer"))

  final val channels = new IntParam(this, "channels", "The number of channels")
  final val sampleRate = new IntParam(this, "sampleRate", "The audio sample rate")
  final val sampleFormat = new IntParam(this, "sampleFormat", "The audio sample format")

  protected var inputColIndex: Int = _

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
   * Sets the channels parameter.
   *
   * @param value the value of the parameter
   */
  def setChannels(value: Int): this.type = set(channels, value)

  /**
   * Sets the sampleRate parameter.
   *
   * @param value the value of the parameter
   */
  def setSampleRate(value: Int): this.type = set(sampleRate, value)

  /**
   * Sets the sampleFormat parameter.
   *
   * @param value the value of the parameter
   */
  def setSampleFormat(value: Int): this.type = set(sampleFormat, value)

  setDefault(outputClass, classOf[String])
  setDefault(translatorFactory, new SpeechRecognitionTranslatorFactory())

  /**
   * Performs speech recognition on the provided dataset.
   *
   * @param dataset input dataset
   * @return output dataset
   */
  def recognize(dataset: Dataset[_]): DataFrame = {
    transform(dataset)
  }

  /** @inheritdoc */
  override def transform(dataset: Dataset[_]): DataFrame = {
    inputColIndex = dataset.schema.fieldIndex($(inputCol))
    super.transform(dataset)
  }

  /**
   * Transforms the rows.
   *
   * @param iter the rows to transform
   * @return the transformed rows
   */
  override protected def transformRows(iter: Iterator[Row]): Iterator[Row] = {
    val predictor = model.newPredictor()
    iter.map(row => {
      val data = row.getAs[Array[Byte]](inputColIndex)
      val audioFactory = AudioFactory.newInstance
      if (isDefined(channels)) {
        audioFactory.setChannels($(channels))
      }
      if (isDefined(sampleRate)) {
        audioFactory.setSampleRate($(sampleRate))
      }
      if (isDefined(sampleFormat)) {
        audioFactory.setSampleFormat($(sampleFormat))
      }
      val audio = audioFactory.fromInputStream(new ByteArrayInputStream(data))
      Row.fromSeq(row.toSeq :+ predictor.predict(audio))
    })
  }

  /** @inheritdoc */
  def validateInputType(schema: StructType): Unit = {
    validateType(schema($(inputCol)), BinaryType)
  }

  /** @inheritdoc */
  override def transformSchema(schema: StructType): StructType = {
    val outputSchema = StructType(schema.fields :+ StructField($(outputCol), StringType))
    outputSchema
  }
}
