/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.examples.inference;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.examples.inference.util.AbstractExample;
import ai.djl.examples.inference.util.Arguments;
import ai.djl.examples.inference.util.BertDataParser;
import ai.djl.examples.inference.util.LogUtils;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.slf4j.Logger;

public final class BertQaInferenceExample extends AbstractExample {

    private static final Logger logger = LogUtils.getLogger(BertQaInferenceExample.class);

    public static void main(String[] args) {
        new BertQaInferenceExample().runExample(args);
    }

    @Override
    public String predict(Arguments args, Metrics metrics, int iteration)
            throws IOException, TranslateException {
        String predictResult = null;

        BertArguments arguments = (BertArguments) args;
        Path modelDir = arguments.getModelDir();
        String modelName = arguments.getModelName();

        // Device is not not required, default device will be used by Model if not provided.
        // Change to a specific device if needed.
        Device device = Device.defaultDevice();
        Model model = Model.newInstance(device);
        model.load(modelDir, modelName);

        QAInput input = new QAInput(arguments);

        BertDataParser parser = model.getArtifact("vocab.json", BertDataParser::parse);

        logger.info("Question: {}", input.getQuestion());
        logger.info("Paragraph: {}", input.getAnswer());

        BertTranslator translator = new BertTranslator(parser);

        try (Predictor<QAInput, String> predictor = model.newPredictor(translator)) {
            predictor.setMetrics(metrics); // Let predictor collect metrics

            for (int i = 0; i < iteration; ++i) {
                predictResult = predictor.predict(input);
                printProgress(iteration, i);
                collectMemoryInfo(metrics);
            }
        }

        model.close();
        return predictResult;
    }

    @Override
    protected Options getOptions() {
        return BertArguments.getOptions();
    }

    @Override
    protected BertArguments parseArguments(CommandLine cmd) {
        return new BertArguments(cmd);
    }

    private static final class QAInput {

        private String question;
        private String answer;
        private int seqLength;

        QAInput(BertArguments arguments) {
            question = arguments.getQuestion();
            answer = arguments.getAnswer();
            seqLength = arguments.getSeqLength();
        }

        public String getQuestion() {
            return question;
        }

        public String getAnswer() {
            return answer;
        }

        public int getSeqLength() {
            return seqLength;
        }
    }

    private static final class BertTranslator implements Translator<QAInput, String> {

        private BertDataParser parser;
        private List<String> tokens;

        BertTranslator(BertDataParser parser) {
            this.parser = parser;
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, QAInput input) {
            // pre-processing - tokenize sentence
            List<String> tokenQ = BertDataParser.tokenizer(input.getQuestion().toLowerCase());
            List<String> tokenA = BertDataParser.tokenizer(input.getAnswer().toLowerCase());
            int validLength = tokenQ.size() + tokenA.size();
            List<Float> tokenTypes =
                    BertDataParser.getTokenTypes(tokenQ, tokenA, input.getSeqLength());
            tokens = BertDataParser.formTokens(tokenQ, tokenA, input.getSeqLength());
            List<Integer> indexes = parser.token2idx(tokens);
            float[] types = Utils.toFloatArray(tokenTypes);
            float[] indexesFloat = Utils.toFloatArray(indexes);

            int seqLength = input.getSeqLength();
            NDManager manager = ctx.getNDManager();
            NDArray data0 = manager.create(indexesFloat, new Shape(1, seqLength));
            NDArray data1 = manager.create(types, new Shape(1, seqLength));
            NDArray data2 = manager.create(new float[] {validLength});

            NDList list = new NDList(3);
            list.add("data0", data0);
            list.add("data1", data1);
            list.add("data2", data2);

            return list;
        }

        @Override
        public String processOutput(TranslatorContext ctx, NDList list) {
            NDArray array = list.get(0);
            NDList output = array.split(2, 2);
            // Get the formatted logits result
            NDArray startLogits = output.get(0).reshape(new Shape(1, -1));
            NDArray endLogits = output.get(1).reshape(new Shape(1, -1));
            // Get Probability distribution
            NDArray startProb = startLogits.softmax(-1);
            NDArray endProb = endLogits.softmax(-1);
            int startIdx = (int) startProb.argmax(1, true).getFloat(0);
            int endIdx = (int) endProb.argmax(1, true).getFloat(0);
            return tokens.subList(startIdx, endIdx + 1).toString();
        }
    }

    public static final class BertArguments extends Arguments {

        private String question;
        private String answer;
        private int seqLength;

        public BertArguments(CommandLine cmd) {
            super(cmd);
            if (cmd.hasOption("question")) {
                question = cmd.getOptionValue("question");
            } else {
                question = "When did BBC Japan start broadcasting?";
            }
            if (cmd.hasOption("answer")) {
                answer = cmd.getOptionValue("answer");
            } else {
                answer =
                        "BBC Japan was a general entertainment Channel.\nWhich operated between December 2004 and April 2006.\nIt ceased operations after its Japanese distributor folded.";
            }
            if (cmd.hasOption("sequenceLength")) {
                seqLength = Integer.parseInt(cmd.getOptionValue("sequenceLength"));
            } else {
                seqLength = 384;
            }
        }

        public static Options getOptions() {
            Options options = Arguments.getOptions();
            options.addOption(
                    Option.builder("q")
                            .longOpt("question")
                            .hasArg()
                            .argName("QUESTION")
                            .desc("Question of the model")
                            .build());
            options.addOption(
                    Option.builder("a")
                            .longOpt("answer")
                            .hasArg()
                            .argName("ANSWER")
                            .desc("Answer paragraph of the model")
                            .build());
            options.addOption(
                    Option.builder("l")
                            .longOpt("sequence-length")
                            .hasArg()
                            .argName("SEQUENCE-LENGTH")
                            .desc("Sequence Length of the paragraph")
                            .build());
            return options;
        }

        public String getQuestion() {
            return question;
        }

        public String getAnswer() {
            return answer;
        }

        public int getSeqLength() {
            return seqLength;
        }
    }
}
