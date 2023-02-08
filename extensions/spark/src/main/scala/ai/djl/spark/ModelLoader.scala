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
package ai.djl.spark

import ai.djl.inference.Predictor
import ai.djl.repository.zoo.{Criteria, ZooModel}
import ai.djl.training.util.ProgressBar
import ai.djl.translate.Translator

import java.io.Serializable

/**
 * `ModelLoader` load [[ai.djl.repository.zoo.ZooModel]] for Spark support.

 * @param url The url of the model
 * @param outputClass The output class
 */
@SerialVersionUID(1L)
class ModelLoader[A, B](val engine: String, val url: String, val inputClass : Class[A], val outputClass : Class[B])
  extends Serializable {

  @transient var model: ZooModel[A, B] = _

  /**
   * Creates a new Predictor.
   *
   * @param translator The translator to use for inference
   * @return an instance of `Predictor`
   */
  def newPredictor(translator: Translator[A, B]): Predictor[A, B] = {
    if (model == null) {
      val criteria = Criteria.builder
        .setTypes(inputClass, outputClass)
        .optEngine(engine)
        .optModelUrls(url)
        .optTranslator(translator)
        .optProgress(new ProgressBar)
        .build
      model = criteria.loadModel
    }
    model.newPredictor
  }
}
