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
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Layout;
import com.amazon.ai.ndarray.types.Shape;
import org.apache.commons.cli.*;
import org.apache.mxnet.engine.MxModel;
import org.apache.mxnet.engine.MxNDArray;
import org.slf4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class BERTQAInferenceExample {
    private static Logger logger = LogUtils.getLogger(BERTQAInferenceExample.class);


    public void runExample(String[] args) {
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

            int iteration = arguments.getIteration();

            logger.info("Running {}, iteration: {}", getClass().getSimpleName(), iteration);

            long init = System.nanoTime();
            String version = Engine.getInstance().getVersion();
            long loaded = System.nanoTime();
            logger.info(
                    String.format(
                            "Load library %s in %.3f ms.", version, (loaded - init) / 1000000f));



            predict(modelDir, modelName, new QAInput(question, answer, seqLength));

        } catch (ParseException e) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.setLeftPadding(1);
            formatter.setWidth(120);
            formatter.printHelp(e.getMessage(), options);
        } catch (Throwable t) {
            logger.error("Unexpected error", t);
        }

    }

    public void predict(String modelDir, String modelName, QAInput input)
    throws IOException {
        String modelPathPrefix = modelDir + '/' + modelName;

        Model model = Model.loadModel(modelPathPrefix, 0);

        DataDesc dataDescs[] = new DataDesc[]{
                new DataDesc(new Shape(1, input.seqLength), DataType.FLOAT32, "data0", Layout.NT),
                new DataDesc(new Shape(1, input.seqLength), DataType.FLOAT32, "data1", Layout.NT),
                new DataDesc(new Shape(1), DataType.FLOAT32, "data2", Layout.NT)
        };

        ((MxModel) model).setDataNames(dataDescs);
    }

    public static void main(String[] args) {
        new BERTQAInferenceExample().runExample(args);
    }

    private static class QAInput {
        String Q;
        String A;
        int seqLength;

        public QAInput(String Q, String A, int seqLength) {
            this.Q = Q;
            this.A = A;
            this.seqLength = seqLength;
        }
    }

    private static class QAOutput { String A; }

    private static final class GenericTranslator implements Translator<QAInput, QAOutput> {

        private BertDataParser util;

        public GenericTranslator(BertDataParser parser) { this.util = parser; }

        @Override
        public NDList processInput(Predictor<?, ?> predictor, QAInput input) {
            // pre-processing - tokenize sentence
            List<String> tokenQ = util.tokenizer(input.Q.toLowerCase());
            List<String> tokenA = util.tokenizer(input.A.toLowerCase());
            int validLength = tokenQ.size() + tokenA.size();
            logger.info("Valid length: " + validLength);
            // generate token types [0000...1111....0000]
            List<Float> QAEmbedded = new ArrayList<>();
            util.pad(QAEmbedded, 0f, tokenQ.size()).addAll(
                    util.pad(new ArrayList<Float>(), 1f, tokenA.size())
            );
            List<Float> tokenTypes = util.pad(QAEmbedded, 0f, input.seqLength);
            // make BERT pre-processing standard
            tokenQ.add("[SEP]");
            tokenQ.add(0, "[CLS]");
            tokenA.add("[SEP]");
            tokenQ.addAll(tokenA);
            List<String> tokens = util.pad(tokenQ, "[PAD]", input.seqLength);
            logger.info("Pre-processed tokens: " + Arrays.toString(tokenQ.toArray()));
            // pre-processing - token to index translation
            List<Integer> indexes = util.token2idx(tokens);
            List<Float> indexesFloat = new ArrayList<>();
            for (int integer : indexes) {
                indexesFloat.add((float) integer);
            }
            Model model = predictor.getModel();
            DataDesc dataDescs[] = model.describeInput();
            predictor.create(dataDescs[0]);
            NDList list = new NDList();
            Arrays.stream(dataDescs).forEach(ele -> list.add(predictor.create(ele)));

            list.get(0).set(indexesFloat);
            list.get(1).set(tokenTypes);
            list.get(2).set(new float[]{validLength});

            return list;
        }

        @Override
        public QAOutput processOutput(Predictor<?, ?> predictor, NDList array) {
            return null;
        }
    }
}
