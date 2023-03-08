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
package ai.djl.spark.translator.binary

import ai.djl.ndarray.NDList
import ai.djl.translate.{Batchifier, Translator, TranslatorContext}

/** A [[ai.djl.translate.Translator]] for Numpy tasks in Spark. */
@SerialVersionUID(1L)
class NpBinaryTranslator(val batchifier: Batchifier) extends Translator[Array[Byte], Array[Byte]]
  with Serializable {

  /** @inheritdoc */
  override def processInput(ctx: TranslatorContext, input: Array[Byte]): NDList = {
    NDList.decode(ctx.getNDManager, input)
  }

  /** @inheritdoc */
  override def processOutput(ctx: TranslatorContext, list: NDList): Array[Byte] = {
    list.encode(true)
  }

  /** @inheritdoc */
  override def getBatchifier: Batchifier = batchifier
}
