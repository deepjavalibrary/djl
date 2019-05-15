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
import com.amazon.ai.engine.Engine;
import com.amazon.ai.example.util.Arguments;
import com.amazon.ai.example.util.BertDataParser;
import com.amazon.ai.example.util.LogUtils;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Layout;
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
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.mxnet.engine.MxModel;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;

public class BertQaInferenceExample {
    private static Logger logger = LogUtils.getLogger(BertQaInferenceExample.class);

    private void runExample(String[] args) {
        Options options = Arguments.getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            Arguments arguments = new Arguments(cmd);

            String modelDir = arguments.getModelDir();
            String modelName = arguments.getModelName();
            String question = arguments.getQuestion();
            String answer = arguments.getAnswer();
            int seqLength = arguments.getSeqLength();
            String vocabulary = modelDir + "/" + arguments.getVocabulary();
            BertDataParser util = new BertDataParser(vocabulary);

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

        Model model = Model.loadModel(modelPathPrefix, 2);

        DataDesc[] dataDescs =
                new DataDesc[] {
                    new DataDesc(
                            new Shape(1, input.seqLength), DataType.FLOAT32, "data0", Layout.NT),
                    new DataDesc(
                            new Shape(1, input.seqLength), DataType.FLOAT32, "data1", Layout.NT),
                    new DataDesc(new Shape(1), DataType.FLOAT32, "data2", Layout.NT),
                };

        ((MxModel) model).setDataNames(dataDescs);

        GenericTranslator translator = new GenericTranslator(parser);
        try (Predictor<QAInput, QAOutput> predictor = Predictor.newInstance(model, translator)) {
            QAOutput output = predictor.predict(input);
            logger.info(
                    String.format(
                            "%nQuestion: %s%nParagraph: %s%nAnswer: %s",
                            input.question, input.answer, output.answer));
        }
    }

    public static void main(String[] args) {
        new BertQaInferenceExample().runExample(args);
    }

    private static class QAInput {
        String question;
        String answer;
        int seqLength;

        QAInput(String question, String answer, int seqLength) {
            this.question = question;
            this.answer = answer;
            this.seqLength = seqLength;
        }
    }

    private static class QAOutput {
        String answer;

        QAOutput(String answer) {
            this.answer = answer;
        }
    }

    private static final class GenericTranslator implements Translator<QAInput, QAOutput> {

        private BertDataParser util;
        private List<String> tokens;

        GenericTranslator(BertDataParser parser) {
            this.util = parser;
        }

        @Override
        public NDList processInput(Predictor<?, ?> predictor, QAInput input) {
            // pre-processing - tokenize sentence
            List<String> tokenQ = util.tokenizer(input.question.toLowerCase());
            List<String> tokenA = util.tokenizer(input.answer.toLowerCase());
            int validLength = tokenQ.size() + tokenA.size();
            List<Float> tokenTypes = util.getTokenTypes(tokenQ, tokenA, input.seqLength);
            tokens = util.formTokens(tokenQ, tokenA, input.seqLength);
            List<Integer> indexes = util.token2idx(tokens);
            List<Float> indexesFloat = new ArrayList<>();
            for (int integer : indexes) {
                indexesFloat.add((float) integer);
            }
            // Start building model
            Model model = predictor.getModel();
            DataDesc[] dataDescs = model.describeInput();
            predictor.create(dataDescs[0]);
            NDList list = new NDList();
            Arrays.stream(dataDescs).forEach(ele -> list.add(predictor.create(ele)));

            list.get(0).set(indexesFloat);
            list.get(1).set(tokenTypes);
            list.get(2).set(Collections.singletonList((float) validLength));

            return list;
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

        @Override
        public QAOutput processOutput(Predictor<?, ?> predictor, NDList list) {
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
            return new QAOutput(tokens.subList(startIdx, endIdx + 1).toString());
        }
    }
}
