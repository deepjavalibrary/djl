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
import com.amazon.ai.Translator;
import com.amazon.ai.TranslatorContext;
import com.amazon.ai.engine.Engine;
import com.amazon.ai.example.util.Arguments;
import com.amazon.ai.example.util.BertDataParser;
import com.amazon.ai.example.util.LogUtils;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Shape;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.mxnet.engine.MxModel;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;

public class BertQaInferenceExample {
    private static Logger logger = LogUtils.getLogger(BertQaInferenceExample.class);

    private void runExample(String[] args) {
        Options options = BertArguments.getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            BertArguments arguments = new BertArguments(cmd);

            String modelDir = arguments.getModelDir();
            String modelName = arguments.getModelName();
            String question = arguments.getQuestion();
            String answer = arguments.getAnswer();
            int seqLength = arguments.getSeqLength();
            String vocabulary = modelDir + "/" + arguments.getVocabulary();
            BertDataParser util = BertDataParser.parse(vocabulary);

            long init = System.nanoTime();
            String version = Engine.getInstance().getVersion();
            Thread.sleep(2000);
            Set<String> set = JnaUtils.getAllOpNames();
            logger.info(set.toString());
            long loaded = System.nanoTime();
            logger.info(
                    String.format(
                            "Load library %s in %.3f ms.", version, (loaded - init) / 1000000f));

            predict(modelDir, modelName, util, new QAInput(question, answer, seqLength));

        } catch (ParseException e) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.setLeftPadding(1);
            formatter.setWidth(120);
            formatter.printHelp(e.getMessage(), options);
        } catch (Throwable t) {
            logger.error("Unexpected error", t);
        }
    }

    private void predict(String modelDir, String modelName, BertDataParser parser, QAInput input)
            throws IOException {
        String modelPathPrefix = modelDir + '/' + modelName;

        Model model = Model.loadModel(modelPathPrefix);

        int seqLength = input.getSeqLength();
        DataDesc data0 = new DataDesc(new Shape(1, seqLength), DataType.FLOAT32, "data0");
        DataDesc data1 = new DataDesc(new Shape(1, seqLength), DataType.FLOAT32, "data1");
        DataDesc data2 = new DataDesc(new Shape(1), DataType.FLOAT32, "data2");
        ((MxModel) model).setDataNames(data0, data1, data2);

        GenericTranslator translator = new GenericTranslator(parser);
        try (Predictor<QAInput, String> predictor = Predictor.newInstance(model, translator)) {
            String answer = predictor.predict(input);
            logger.info("Question: {}", input.getQuestion());
            logger.info("Paragraph: {}", input.getAnswer());
            logger.info("Answer: {}", answer);
        }
    }

    public static void main(String[] args) {
        new BertQaInferenceExample().runExample(args);
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

    private static final class GenericTranslator implements Translator<QAInput, String> {

        private BertDataParser parser;
        private List<String> tokens;

        GenericTranslator(BertDataParser parser) {
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
            // Start building model
            Model model = ctx.getModel();
            DataDesc[] dataDescs = model.describeInput();
            NDFactory factory = ctx.getNDFactory();
            factory.create(dataDescs[0]);
            NDList list = new NDList(3);
            Arrays.stream(dataDescs).forEach(ele -> list.add(factory.create(ele)));

            list.get(0).set(indexesFloat);
            list.get(1).set(tokenTypes);
            list.get(2).set(Collections.singletonList((float) validLength));

            return list;
        }

        @Override
        public String processOutput(TranslatorContext ctx, NDList list) {
            NDArray array = list.get(0);
            NDArray[] output = array.split(2, 2, null);
            // Get the formatted logits result
            NDArray startLogits = output[0].reshape(0, -3);
            NDArray endLogits = output[1].reshape(0, -3);
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
        private String vocabulary;

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
            if (cmd.hasOption("vocabulary")) {
                vocabulary = cmd.getOptionValue("vocabulary");
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
            options.addOption(
                    Option.builder("v")
                            .longOpt("vocabulary")
                            .hasArg()
                            .argName("VOCABULARY")
                            .desc("Vocabulary of the model")
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

        public String getVocabulary() {
            return vocabulary;
        }
    }
}
