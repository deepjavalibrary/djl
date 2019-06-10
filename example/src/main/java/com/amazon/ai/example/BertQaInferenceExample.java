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

package com.amazon.ai.example;

import com.amazon.ai.Model;
import com.amazon.ai.TranslateException;
import com.amazon.ai.Translator;
import com.amazon.ai.TranslatorContext;
import com.amazon.ai.example.util.AbstractExample;
import com.amazon.ai.example.util.Arguments;
import com.amazon.ai.example.util.BertDataParser;
import com.amazon.ai.example.util.LogUtils;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.metric.Metrics;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.Shape;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.slf4j.Logger;

public final class BertQaInferenceExample extends AbstractExample {

    private static Logger logger = LogUtils.getLogger(BertQaInferenceExample.class);

    private BertQaInferenceExample() {}

    public static void main(String[] args) {
        new BertQaInferenceExample().runExample(args);
    }

    @Override
    public void predict(Arguments args, int iteration) throws IOException, TranslateException {
        BertArguments arguments = (BertArguments) args;

        File modelDir = new File(arguments.getModelDir());
        String modelName = arguments.getModelName();

        Model model = Model.loadModel(modelDir, modelName);

        String question = arguments.getQuestion();
        String answer = arguments.getAnswer();
        int seqLength = arguments.getSeqLength();

        BertDataParser parser = model.getArtifact("vocab.json", BertDataParser::parse);
        QAInput input = new QAInput(question, answer, seqLength);

        logger.info("Question: {}", input.getQuestion());
        logger.info("Paragraph: {}", input.getAnswer());

        BertTranslator translator = new BertTranslator(parser);
        Metrics metrics = new Metrics();

        try (Predictor<QAInput, String> predictor = Predictor.newInstance(model, translator)) {
            predictor.setMetrics(metrics);

            for (int i = 0; i < iteration; ++i) {
                String result = predictor.predict(input);
                printProgress(iteration, i, result);
            }

            float p50 = metrics.percentile("Inference", 50).getValue().longValue() / 1000000f;
            float p90 = metrics.percentile("Inference", 90).getValue().longValue() / 1000000f;

            logger.info(String.format("inference P50: %.3f ms, P90: %.3f ms", p50, p90));

            dumpMemoryInfo(metrics, args.getLogDir());
        }
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

        QAInput(String question, String answer, int seqLength) {
            this.question = question;
            this.answer = answer;
            this.seqLength = seqLength;
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
            List<Float> indexesFloat = new ArrayList<>(indexes.size());
            for (int integer : indexes) {
                indexesFloat.add((float) integer);
            }

            int seqLength = input.getSeqLength();
            NDFactory factory = ctx.getNDFactory();
            NDArray data0 = factory.create(new DataDesc(new Shape(1, seqLength)));
            NDArray data1 = factory.create(new DataDesc(new Shape(1, seqLength)));
            NDArray data2 = factory.create(new DataDesc(new Shape(1)));

            data0.set(indexesFloat);
            data1.set(tokenTypes);
            data2.set(Collections.singletonList((float) validLength));

            NDList list = new NDList(3);
            list.add("data0", data0);
            list.add("data1", data1);
            list.add("data2", data2);

            return list;
        }

        @Override
        public String processOutput(TranslatorContext ctx, NDList list) {
            NDArray array = list.get(0);
            NDList output = array.split(2, 2, null);
            // Get the formatted logits result
            NDArray startLogits = output.get(0).reshape(0, -3);
            NDArray endLogits = output.get(1).reshape(0, -3);
            // Get Probability distribution
            float[] startProb = startLogits.softmax(null, null).toFloatArray();
            float[] endProb = endLogits.softmax(null, null).toFloatArray();
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
            }
            if (cmd.hasOption("answer")) {
                answer = cmd.getOptionValue("answer");
            }
            if (cmd.hasOption("sequenceLength")) {
                seqLength = Integer.parseInt(cmd.getOptionValue("sequenceLength"));
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
                    Option.builder("sl")
                            .longOpt("sequenceLength")
                            .hasArg()
                            .argName("SEQUENCELENGTH")
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
