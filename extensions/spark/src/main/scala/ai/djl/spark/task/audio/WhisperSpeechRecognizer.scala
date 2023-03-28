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

import ai.djl.audio.translator.WhisperTranslatorFactory
import ai.djl.modality.audio.Audio
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.bytedeco.javacv.{FFmpegFrameGrabber, Frame, FrameGrabber}

import java.io.{ByteArrayInputStream, IOException}
import java.nio.ShortBuffer
import java.util
import scala.jdk.CollectionConverters.asScalaBufferConverter

/**
 * WhisperSpeechRecognizer is very similar to the SpeechRecognizer that performs speech recognition on audio,
 * except that this API is specially tailored for OpenAI Whisper related models.
 *
 * @param uid An immutable unique ID for the object and its derivatives.
 */
class WhisperSpeechRecognizer(override val uid: String) extends SpeechRecognizer {

  def this() = this(Identifiable.randomUID("WhisperSpeechRecognizer"))

  private final val CHUNK_LENGTH: Int = 200

  setDefault(translatorFactory, new WhisperTranslatorFactory())

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
      val audioList = splitAudio(data)
      val result = predictor.batchPredict(audioList)
      Row.fromSeq(row.toSeq :+ String.join("", result))
    })
  }

  /** @inheritdoc */
  override def transformSchema(schema: StructType): StructType = {
    val outputSchema = StructType(schema.fields :+ StructField($(outputCol), StringType))
    outputSchema
  }

  /**
   * Split the audio data into chunks.
   *
   * @param data the audio data
   * @return the list of audio chunks
   */
  private def splitAudio(data: Array[Byte]): util.ArrayList[Audio] = {
    val audioList = new util.ArrayList[Audio]()
    val grabber = new FFmpegFrameGrabber(new ByteArrayInputStream(data))
    try {
      if (isDefined(channels)) {
        grabber.setAudioChannels($(channels))
      }
      if (isDefined(sampleRate)) {
        grabber.setSampleRate($(sampleRate))
      }
      if (isDefined(sampleFormat)) {
        grabber.setSampleFormat($(sampleFormat))
      }
      grabber.start()
      val list = new util.ArrayList[Float]()
      var frame: Frame = null
      var i = 0
      while ( {
        frame = grabber.grabFrame(true, false, true, false, false)
        frame != null
      }) {
        if (i > CHUNK_LENGTH) {
          audioList.add(new Audio(list.asScala.toArray, grabber.getSampleRate, grabber.getAudioChannels))
          list.clear()
          i = 0 // Reset the counter
        }
        val buf = frame.samples(0).asInstanceOf[ShortBuffer]
        for (j <- 0 until buf.limit) {
          list.add(buf.get / Short.MaxValue.toFloat)
        }
        i += 1
      }
      if (!list.isEmpty) {
        audioList.add(new Audio(list.asScala.toArray, grabber.getSampleRate, grabber.getAudioChannels))
      }
      audioList
    } catch {
      case e: FrameGrabber.Exception =>
        throw new IOException("Unsupported Audio file", e)
    } finally {
      if (grabber != null) {
        grabber.close()
      }
    }
  }
}