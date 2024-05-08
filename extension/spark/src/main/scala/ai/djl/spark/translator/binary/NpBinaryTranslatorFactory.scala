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

import ai.djl.Model
import ai.djl.translate.{ArgumentsUtil, Batchifier, Translator, TranslatorFactory}
import ai.djl.util.Pair

import java.lang.reflect.Type
import java.util

/** A [[{ai.djl.translate.TranslatorFactory]]} that creates a [[NpBinaryTranslator]] instance. */
@SerialVersionUID(1L)
class NpBinaryTranslatorFactory extends TranslatorFactory with Serializable {

  /** @inheritdoc */
  override def getSupportedTypes: util.Set[Pair[Type, Type]] = {
    val supportedTypes = new util.HashSet[Pair[Type, Type]]
    supportedTypes.add(new Pair[Type, Type](classOf[Array[Byte]], classOf[Array[Byte]]))
    supportedTypes
  }

  /** @inheritdoc */
  override def newInstance[I, O](input: Class[I], output: Class[O], model: Model,
                                 arguments: java.util.Map[String, _]): Translator[I, O] = {
    val batchifier = ArgumentsUtil.stringValue(arguments, "batchifier", "none")
    if ((input eq classOf[Array[Byte]]) && (output eq classOf[Array[Byte]])) {
      return new NpBinaryTranslator(Batchifier.fromString(batchifier)).asInstanceOf[Translator[I, O]]
    }
    throw new IllegalArgumentException("Unsupported input/output types.")
  }
}
