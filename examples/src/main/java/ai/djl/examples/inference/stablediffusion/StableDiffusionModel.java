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
package ai.djl.examples.inference.stablediffusion;

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.util.Arrays;

public class StableDiffusionModel {

    private static final int HEIGHT = 512;
    private static final int WIDTH = 512;
    private static final int OFFSET = 1;
    private static final float GUIDANCE_SCALE = 7.5f;
    private static final float STRENGTH = 0.75f;

    private Predictor<Image, NDArray> vaeEncoder;
    private Predictor<NDArray, Image> vaeDecoder;
    private Predictor<String, NDList> textEncoder;
    private Predictor<NDList, NDList> unetExecutor;
    private Device device;

    public StableDiffusionModel(Device device) throws ModelException, IOException {
        this.device = device;
        String type = device.getDeviceType();
        if (!"cpu".equals(type) && !"gpu".equals(type)) {
            throw new UnsupportedOperationException(type + " device not supported!");
        }
        Criteria<NDList, NDList> unetCriteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(
                                "https://resources.djl.ai/demo/pytorch/stable-diffusion/"
                                        + type
                                        + "/unet_traced_model.zip")
                        .optEngine("PyTorch")
                        .optProgress(new ProgressBar())
                        .optTranslator(new NoopTranslator())
                        .optDevice(device)
                        .build();
        this.unetExecutor = unetCriteria.loadModel().newPredictor();
        Criteria<NDArray, Image> decoderCriteria =
                Criteria.builder()
                        .setTypes(NDArray.class, Image.class)
                        .optModelUrls(
                                "https://resources.djl.ai/demo/pytorch/stable-diffusion/"
                                        + type
                                        + "/vae_decode_model.zip")
                        .optEngine("PyTorch")
                        .optTranslator(new ImageDecoder())
                        .optProgress(new ProgressBar())
                        .optDevice(device)
                        .build();
        this.vaeDecoder = decoderCriteria.loadModel().newPredictor();
        Criteria<String, NDList> criteria =
                Criteria.builder()
                        .setTypes(String.class, NDList.class)
                        .optModelUrls(
                                "https://resources.djl.ai/demo/pytorch/stable-diffusion/"
                                        + type
                                        + "/text_encoder.zip")
                        .optEngine("PyTorch")
                        .optProgress(new ProgressBar())
                        .optTranslator(new TextEncoder())
                        .optDevice(device)
                        .build();
        this.textEncoder = criteria.loadModel().newPredictor();
    }

    public Image generateImageFromText(String prompt, int steps)
            throws ModelException, IOException, TranslateException {
        return generateImageFromImage(prompt, null, steps);
    }

    public Image generateImageFromImage(String prompt, Image image, int steps)
            throws ModelException, IOException, TranslateException {
        // TODO: implement this part
        try (NDManager manager = NDManager.newBaseManager(device, "PyTorch")) {
            // Step 1: Build text embedding
            NDList textEncoding = textEncoder.predict(prompt);
            NDList uncondEncoding = textEncoder.predict("");
            textEncoding.attach(manager);
            uncondEncoding.attach(manager);
            NDArray textEncodingArray = textEncoding.get(1);
            NDArray uncondEncodingArray = uncondEncoding.get(1);
            NDArray embeddings = textEncodingArray.concat(uncondEncodingArray);
            // Step 2: Build latent
            PndmScheduler scheduler = new PndmScheduler(manager);
            scheduler.initTimesteps(steps, OFFSET);
            Shape latentInitShape = new Shape(1, 4, HEIGHT / 8, WIDTH / 8);
            NDArray latent;
            if (image != null) {
                loadImageEncoder();
                latent = vaeEncoder.predict(image);
                NDArray noise = manager.randomNormal(latent.getShape());
                // Step 2.5: reset timestep to reflect on the given image
                int initTimestep = (int) (steps * STRENGTH) + OFFSET;
                initTimestep = Math.min(initTimestep, steps);
                int[] timestepsArr = scheduler.getTimesteps();
                int timesteps = timestepsArr[timestepsArr.length - initTimestep];
                latent = scheduler.addNoise(latent, noise, timesteps);
                int tStart = Math.max(steps - initTimestep + OFFSET, 0);
                scheduler.setTimesteps(
                        Arrays.copyOfRange(timestepsArr, tStart, timestepsArr.length));
            } else {
                latent = manager.randomNormal(latentInitShape);
            }
            // Step 3: Start iterating/generating
            ProgressBar pb = new ProgressBar("Generating", steps);
            pb.start(0);
            for (int i = 0; i < scheduler.getTimesteps().length; i++) {
                long t = scheduler.getTimesteps()[i];
                NDArray latentModelOutput = latent.concat(latent);
                NDArray noisePred =
                        unetExecutor
                                .predict(
                                        new NDList(
                                                latentModelOutput, manager.create(t), embeddings))
                                .get(0);
                NDList splitNoisePred = noisePred.split(2);
                NDArray noisePredText = splitNoisePred.get(0);
                NDArray noisePredUncond = splitNoisePred.get(1);
                NDArray scaledNoisePredUncond = noisePredText.add(noisePredUncond.neg());
                scaledNoisePredUncond = scaledNoisePredUncond.mul(GUIDANCE_SCALE);
                noisePred = noisePredUncond.add(scaledNoisePredUncond);
                latent = scheduler.step(noisePred, (int) t, latent);
                pb.increment(1);
            }
            pb.end();
            // Step 4: get final image
            return vaeDecoder.predict(latent);
        }
    }

    private void loadImageEncoder() throws ModelException, IOException {
        if (vaeEncoder != null) {
            return;
        }
        Criteria<Image, NDArray> criteria =
                Criteria.builder()
                        .setTypes(Image.class, NDArray.class)
                        .optModelUrls(
                                "https://resources.djl.ai/demo/pytorch/stable-diffusion/"
                                        + device.getDeviceType()
                                        + "/vae_encode_model.zip")
                        .optEngine("PyTorch")
                        .optTranslator(new ImageEncoder(HEIGHT, WIDTH))
                        .optProgress(new ProgressBar())
                        .optDevice(device)
                        .build();
        vaeEncoder = criteria.loadModel().newPredictor();
    }
}
