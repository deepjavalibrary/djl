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
package ai.djl.spark.translator.text

import ai.djl.ndarray.NDList
import ai.djl.translate.{Batchifier, Translator, TranslatorContext}

/** A [[ai.djl.translate.Translator]] for Text Embedding tasks in Spark. */
@SerialVersionUID(1L)
class TextEmbeddingTranslator extends Translator[String, Array[Float]] with Serializable {

  /** @inheritdoc */
  override def processInput(ctx: TranslatorContext, input: String): NDList = {
    new NDList(ctx.getNDManager.create(input).expandDims(0))
  }

  /** @inheritdoc */
  override def processOutput(ctx: TranslatorContext, list: NDList): Array[Float] = {
    list.singletonOrThrow.get(0).toFloatArray
  }

  /** @inheritdoc */
  override def getBatchifier: Batchifier = null
}
