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
package ai.djl.examples.inference.nlp;

import ai.djl.ModelException;
import ai.djl.testing.TestRequirements;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class TextGenerationTest {

    @Test
    public void testTextGeneration() throws TranslateException, ModelException, IOException {
        TestRequirements.linux();
        TestRequirements.weekly();

        // Greedy
        String expected =
                "DeepMind Company is a global leader in the field of artificial"
                        + " intelligence and artificial intelligence. We are a leading provider"
                        + " of advanced AI solutions for the automotive industry, including the"
                        + " latest in advanced AI solutions for the automotive industry. We are"
                        + " also a leading provider of advanced AI solutions for the automotive"
                        + " industry, including the";
        Assert.assertEquals(TextGeneration.generateTextWithPyTorchGreedy(), expected);

        // Contrastive
        String[] output1 = TextGeneration.generateTextWithPyTorchContrastive();
        Assert.assertEquals(
                output1[0],
                "DeepMind Company is a leading provider of advanced AI solutions for businesses,"
                        + " government agencies and individuals. We offer a wide range of services"
                        + " including research, development, training, consulting, and"
                        + " support.<|endoftext|>This article is about the character. You may be"
                        + " looking for the original version");
        Assert.assertEquals(
                output1[1],
                "Memories follow me left and right. I can't remember the last time I saw her.\n"
                        + "\n"
                        + "\"What do you mean?\" asked my mother.\n"
                        + "\n"
                        + "\"I'm sorry, but I don't know what happened to her.\"\n"
                        + "\n"
                        + "\"Well, you're right. She was very");

        // Beam
        String[] output2 = TextGeneration.generateTextWithPyTorchBeam();
        Assert.assertEquals(
                output2[0],
                "DeepMind Company is a global leader in the field of artificial intelligence and"
                    + " artificial intelligence research and development.\n"
                    + "\n"
                    + "Our mission is to provide the world with the best and brightest minds in the"
                    + " field of artificial intelligence and artificial intelligence research and"
                    + " development.\n"
                    + "\n"
                    + "Our mission is to");
        Assert.assertTrue(
                output2[3].startsWith(
                        "Memories follow me left and right. I can't tell you how many times I've"
                                + " been told that I'm not a good person."));
    }

    @Test
    public void testSeqBatchScheduler() throws TranslateException, ModelException, IOException {
        TestRequirements.linux();
        TestRequirements.weekly();
        String[] output = RollingBatch.seqBatchSchedulerWithPyTorchContrastive();
        Assert.assertEquals(
                output[0],
                "DeepMind Company is a leading provider of advanced AI solutions for businesses,"
                        + " government agencies and individuals. We offer a wide range of services"
                        + " including research, development");
        Assert.assertEquals(
                output[1],
                "Memories follow me left and right. I can't wait to see what happens next.\n"
                        + "\n"
                        + "Advertisements<|endoftext|>");
        Assert.assertEquals(
                output[2],
                "When your legs don't work like they used to before And I can't sweep you off my"
                        + " feet, but I can help you out with your hair");
        Assert.assertEquals(
                output[3],
                "There's a time that I remember, when I did not know what to do with myself. I felt"
                        + " like I was going to die. I thought");
        Assert.assertEquals(
                output[4],
                "A person gets sent back to prison for life.\n"
                        + "\n"
                        + "But if you're lucky, you can escape from prison and live happily ever"
                        + " after.\n");
    }

    @Test
    public void testTextGenerationWithOnnx()
            throws TranslateException, ModelException, IOException {
        TestRequirements.linux();
        TestRequirements.weekly();
        TestRequirements.engine("PyTorch");

        // Beam with Ort
        String[] output0 = TextGeneration.generateTextWithOnnxRuntimeBeam();
        Assert.assertEquals(
                output0[0],
                "DeepMind Company is a global leader in the field of artificial intelligence and"
                    + " artificial intelligence research and development.\n"
                    + "\n"
                    + "Our mission is to provide the world with the best and brightest minds in the"
                    + " field of artificial intelligence and artificial intelligence research and"
                    + " development.\n"
                    + "\n"
                    + "Our mission is to provide the world with the best");
    }
}
