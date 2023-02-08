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
package ai.djl.spark.translator.vision

import ai.djl.modality.Classifications
import ai.djl.modality.cv.transform.{Resize, ToTensor}
import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.translate.{Batchifier, Pipeline, Translator, TranslatorContext}
import ai.djl.util.Utils
import org.apache.spark.ml.image.ImageSchema
import org.apache.spark.sql.Row

import java.util

/** A [[ai.djl.translate.Translator]] for Spark Image Classification tasks. */
@SerialVersionUID(1L)
class ImageClassificationTranslator extends Translator[Row, Classifications] with Serializable {

  private var classes: util.List[String] = new util.ArrayList[String]()
  @transient private lazy val pipeline = new Pipeline()
    .add(new Resize(224, 224))
    .add(new ToTensor())

  /** @inheritdoc */
  override def prepare(ctx: TranslatorContext): Unit = {
    classes = Utils.readLines(ctx.getModel.getArtifact("synset.txt").openStream())
  }

  /** @inheritdoc */
  override def processInput(ctx: TranslatorContext, input: Row): NDList = {
    val height = ImageSchema.getHeight(input)
    val width = ImageSchema.getWidth(input)
    val channel = ImageSchema.getNChannels(input)
    var image = ctx.getNDManager
      .create(ImageSchema.getData(input), new Shape(height, width, channel))
      .toType(DataType.UINT8, true)
    // BGR to RGB
    image = image.flip(2)
    pipeline.transform(new NDList(image))
  }

  /** @inheritdoc */
  override def processOutput(ctx: TranslatorContext, list: NDList): Classifications = {
    var probabilitiesNd = list.singletonOrThrow
    probabilitiesNd = probabilitiesNd.softmax(0)
    new Classifications(classes, probabilitiesNd)
  }

  /** @inheritdoc */
  override def getBatchifier: Batchifier = Batchifier.STACK
}
