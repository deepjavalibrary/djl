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

package software.amazon.ai.example;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.slf4j.Logger;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.TranslateException;
import software.amazon.ai.Translator;
import software.amazon.ai.TranslatorContext;
import software.amazon.ai.example.util.AbstractExample;
import software.amazon.ai.example.util.Arguments;
import software.amazon.ai.example.util.BertDataParser;
import software.amazon.ai.example.util.LogUtils;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDFactory;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.Utils;

public final class BertQaInferenceExample extends AbstractExample {

    private static Logger logger = LogUtils.getLogger(BertQaInferenceExample.class);

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

        Model model = Model.loadModel(modelDir, modelName);

        QAInput input = new QAInput(arguments);
        BertDataParser parser = model.getArtifact("vocab.json", BertDataParser::parse);

        logger.info("Question: {}", input.getQuestion());
        logger.info("Paragraph: {}", input.getAnswer());

        BertTranslator translator = new BertTranslator(parser);

        // Following context is not not required, default context will be used by Predictor without
        // passing context to Predictor.newInstance(model, translator)
        // Change to a specific context if needed.
        Context context = Context.defaultContext();

        try (Predictor<QAInput, String> predictor =
                Predictor.newInstance(model, translator, context)) {
            predictor.setMetrics(metrics); // Let predictor collect metrics

            for (int i = 0; i < iteration; ++i) {
                predictResult = predictor.predict(input);
                printProgress(iteration, i);
                collectMemoryInfo(metrics);
            }
        }
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
            NDFactory factory = ctx.getNDFactory();
            NDArray data0 = factory.create(new DataDesc(new Shape(1, seqLength), DataType.INT32));
            NDArray data1 = factory.create(new DataDesc(new Shape(1, seqLength)));
            NDArray data2 = factory.create(new DataDesc(new Shape(1), DataType.INT32));

            data0.set(indexesFloat);
            data1.set(types);
            data2.set(new float[] {validLength});

            NDList list = new NDList(3);
            list.add("data0", data0.asType(DataType.FLOAT32, false));
            list.add("data1", data1);
            list.add("data2", data2.asType(DataType.FLOAT32, false));

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
            float[] startProb = startLogits.softmax(-1).toFloatArray();
            float[] endProb = endLogits.softmax(-1).toFloatArray();
            int startIdx = argmax(startProb);
            int endIdx = argmax(endProb);
            return tokens.subList(startIdx, endIdx + 1).toString();
        }

        private static int argmax(float[] prob) {
            int maxIdx = 0;
            for (int i = 0; i < prob.length; i++) {
                if (prob[maxIdx] < prob[i]) {
                    maxIdx = i;
                }
            }
            return maxIdx;
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
