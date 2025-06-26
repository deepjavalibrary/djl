/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.inference.whisper;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.audio.Audio;
import ai.djl.modality.audio.AudioFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.DownloadUtils;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URL;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

public final class WhisperJet {

    private static final Logger logger = LoggerFactory.getLogger(WhisperJet.class);

    private WhisperJet() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        String output = predict();
        logger.info(output);
    }

    public static String predict() throws ModelException, IOException, TranslateException {
        if ("aarch64".equals(System.getProperty("os.arch"))) {
            throw new AssertionError("OnnxRuntime extension 0.13.0 doesn't support apple silicon");
        }

        Criteria<AudioInput, String> criteria =
                Criteria.builder()
                        .setTypes(AudioInput.class, String.class)
                        .optModelUrls(
                                "https://resources.djl.ai/test-models/onnxruntime/WhisperJET.zip")
                        .optModelName("WhisperJET_encoder")
                        .optEngine("OnnxRuntime") // use OnnxRuntime engine
                        .optOption("cpuArenaAllocator", "true")
                        .optOption("memoryPatternOptimization", "true")
                        .optOption("optLevel", "EXTENDED_OPT")
                        .optTranslator(new WhisperJetTranslator())
                        .build();

        String url = "https://resources.djl.ai/audios/testEN.wav";
        Path file = Paths.get("build/tmp/testEn.wav");
        DownloadUtils.download(new URL(url), file, new ProgressBar());
        Audio audio = AudioFactory.newInstance().fromFile(file);
        AudioInput input = new AudioInput(audio, "en");
        try (ZooModel<AudioInput, String> model = criteria.loadModel();
                Predictor<AudioInput, String> predictor = model.newPredictor()) {
            return predictor.predict(input);
        }
    }

    public static final class WhisperJetTranslator
            implements NoBatchifyTranslator<AudioInput, String> {

        private static final int MAX_TOKENS = 445;

        private static final String[] LANGUAGES = {
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar",
            "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu",
            "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa",
            "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn",
            "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
            "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
            "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw",
            "su", "yue"
        };

        private static final int START_TOKEN_ID = 50258;
        private static final int TRANSCRIBE_TOKEN_ID = 50359;
        private static final int NO_TIMESTAMPS_TOKEN_ID = 50363;
        private static final int EOS = 50257;

        private static final int MAX_TOKENS_PER_SECOND = 30;
        private static final float SAMPLE_RATE_CANDIDATES = 16000f;
        private static final float RATIO = MAX_TOKENS_PER_SECOND / SAMPLE_RATE_CANDIDATES;

        private static final Pattern TIMESTAMP_PATTERN = Pattern.compile("<\\|[^>]*\\|> ");

        private Predictor<NDList, NDList> preprocessor;
        private Predictor<NDList, NDList> decoder;
        private Predictor<NDList, NDList> cacheInitializer;
        private Predictor<NDList, NDList> detokenizer;

        @Override
        public void prepare(TranslatorContext ctx) throws ModelException, IOException {
            Model model = ctx.getModel();
            NDManager manager = ctx.getPredictorManager();
            Path modelPath = model.getModelPath();

            ZooModel<NDList, NDList> initModel = loadChildModel(modelPath, "WhisperJET_init");
            preprocessor = initModel.newPredictor();
            manager.attachInternal(NDManager.nextUid(), initModel);
            manager.attachInternal(NDManager.nextUid(), preprocessor);

            ZooModel<NDList, NDList> cacheInitModel =
                    loadChildModel(modelPath, "WhisperJET_cache_initializer");
            cacheInitializer = cacheInitModel.newPredictor();
            manager.attachInternal(NDManager.nextUid(), cacheInitModel);
            manager.attachInternal(NDManager.nextUid(), cacheInitializer);

            ZooModel<NDList, NDList> decoderModel = loadChildModel(modelPath, "WhisperJET_decoder");
            decoder = decoderModel.newPredictor();
            manager.attachInternal(NDManager.nextUid(), decoderModel);
            manager.attachInternal(NDManager.nextUid(), decoder);

            ZooModel<NDList, NDList> detokenizerModel =
                    loadChildModel(modelPath, "WhisperJET_detokenizer");
            detokenizer = detokenizerModel.newPredictor();
            manager.attachInternal(NDManager.nextUid(), detokenizerModel);
            manager.attachInternal(NDManager.nextUid(), detokenizer);
        }

        @Override
        public NDList processInput(TranslatorContext ctx, AudioInput input) throws Exception {
            int languageId = getLanguageID(input.language);
            int[] decoderInitialInputIds = {
                START_TOKEN_ID, languageId, TRANSCRIBE_TOKEN_ID, NO_TIMESTAMPS_TOKEN_ID
            };
            ctx.setAttachment("ids", decoderInitialInputIds);

            float[] data = input.audio.getData();
            FloatBuffer fb = FloatBuffer.wrap(data);
            if (data.length > 16000 * 30) {
                // trim to 30 seconds
                fb.limit(16000 * 30);
            }

            // if we generate more than this number of tokens it means that we have an infinite
            // loop due to the fact that the sound cannot be transcribed with the language
            // selected
            int kout = (int) (data.length * RATIO);
            int maxTokens = Math.min(kout, MAX_TOKENS);
            ctx.setAttachment("maxTokens", maxTokens);

            NDArray array = ctx.getNDManager().create(fb, new Shape(1, fb.remaining()));
            array.setName("fast_pcm");

            NDList inputFeatures = preprocessor.predict(new NDList(array));
            inputFeatures.get(0).setName("input_features");
            return inputFeatures;
        }

        @Override
        public String processOutput(TranslatorContext ctx, NDList list) throws TranslateException {
            int[] decoderInitialInputIds = (int[]) ctx.getAttachment("ids");
            int maxTokens = (int) ctx.getAttachment("maxTokens");

            // initialize decoder cache
            list.get(0).setName("encoder_hidden_states");
            NDList initResult = cacheInitializer.predict(list);
            NDList result = null;

            boolean execution1HitMaxLength = false;

            // DECODER ITERATION
            List<Integer> completeOutput = new ArrayList<>();
            NDManager manager = ctx.getNDManager();
            NDArray inputIds = null;

            long max = -1;
            boolean isFirstIteration = true;
            int count = 1;
            while (max != EOS) {
                if (count <= 4) {
                    long[][] ids = {{decoderInitialInputIds[count - 1]}};
                    inputIds = manager.create(ids);
                    inputIds.setName("input_ids");
                }

                NDList pastValue = new NDList();
                pastValue.add(inputIds);

                if (isFirstIteration) {
                    for (int i = 0; i < 12; i++) {
                        NDArray array = manager.zeros(new Shape(1, 12, 0, 64));
                        array.setName("idk" + i);
                        pastValue.add(array);

                        array = manager.zeros(new Shape(1, 12, 0, 64));
                        array.setName("idv" + i);
                        pastValue.add(array);

                        array = initResult.get("opek" + i);
                        array.setName("iek" + i);
                        pastValue.add(array);

                        array = initResult.get("opev" + i);
                        array.setName("iev" + i);
                        pastValue.add(array);
                    }
                    isFirstIteration = false;
                } else {
                    for (int i = 0; i < 12; i++) {
                        NDArray array = result.get("opdk" + i);
                        array.setName("idk" + i);
                        pastValue.add(array);

                        array = result.get("opdv" + i);
                        array.setName("idv" + i);
                        pastValue.add(array);

                        // array name has been changed already
                        pastValue.add(initResult.get("iek" + i));
                        pastValue.add(initResult.get("iev" + i));
                    }
                }

                NDList nextResult = decoder.predict(pastValue);
                if (result != null) {
                    result.close(); // manually close
                }
                result = nextResult;
                inputIds.close(); // manually close

                NDArray logits = result.get("logits").get(0, 0);
                max = logits.argMax().getLong();
                completeOutput.add(Math.toIntExact(max));

                inputIds = manager.create(new long[][] {{max}});
                inputIds.setName("input_ids");

                if (count >= maxTokens) {
                    execution1HitMaxLength = true;
                    max = EOS;
                }
                count++;
            }

            int[] tokenSequences = completeOutput.stream().mapToInt(i -> i).toArray();
            NDArray array = manager.create(tokenSequences, new Shape(1, 1, tokenSequences.length));
            array.setName("sequences");
            NDArray textOutput = detokenizer.predict(new NDList(array)).get(0);

            String ret = textOutput.toStringArray()[0];
            ret = TIMESTAMP_PATTERN.matcher(ret).replaceAll("");
            ret = ret.trim();

            if (execution1HitMaxLength) {
                ret += " ...";
            }
            return ret;
        }

        private ZooModel<NDList, NDList> loadChildModel(Path modelPath, String modelName)
                throws ModelException, IOException {
            Criteria<NDList, NDList> criteria =
                    Criteria.builder()
                            .setTypes(NDList.class, NDList.class)
                            .optModelPath(modelPath)
                            .optModelName(modelName)
                            .optEngine("OnnxRuntime")
                            .optOption("cpuArenaAllocator", "true")
                            .optOption("memoryPatternOptimization", "true")
                            .optOption("optLevel", "EXTENDED_OPT")
                            .optTranslator(new NoopTranslator())
                            .build();
            return criteria.loadModel();
        }

        private int getLanguageID(String language) {
            for (int i = 0; i < LANGUAGES.length; i++) {
                if (LANGUAGES[i].equals(language)) {
                    return START_TOKEN_ID + i + 1;
                }
            }
            return -1;
        }
    }

    public static final class AudioInput {

        Audio audio;
        String language;

        public AudioInput(Audio audio, String language) {
            this.audio = audio;
            this.language = language;
        }
    }
}
