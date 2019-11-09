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

import ai.djl.ModelException;
import ai.djl.examples.inference.util.AbstractInference;
import ai.djl.examples.inference.util.Arguments;
import ai.djl.examples.util.MemoryUtils;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.mxnet.zoo.nlp.qa.QAInput;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class BertQaInferenceExample extends AbstractInference<String> {

    private static final Logger logger = LoggerFactory.getLogger(BertQaInferenceExample.class);

    public static void main(String[] args) {
        new BertQaInferenceExample().runExample(args);
    }

    /** {@inheritDoc} */
    @Override
    public String predict(Arguments args, Metrics metrics, int iteration)
            throws IOException, TranslateException, ModelException {
        BertArguments arguments = (BertArguments) args;

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("backbone", "bert");
        criteria.put("dataset", "book_corpus_wiki_en_uncased");
        ZooModel<QAInput, String> model = MxModelZoo.BERT_QA.loadModel(criteria, new ProgressBar());

        QAInput input = new QAInput(arguments.question, arguments.answer, arguments.seqLength);

        logger.info("Question: {}", input.getQuestion());
        logger.info("Paragraph: {}", input.getAnswer());

        String predictResult = null;
        try (Predictor<QAInput, String> predictor = model.newPredictor()) {
            predictor.setMetrics(metrics); // Let predictor collect metrics

            for (int i = 0; i < iteration; ++i) {
                predictResult = predictor.predict(input);

                progressBar.update(i);
                MemoryUtils.collectMemoryInfo(metrics);
            }
        }

        model.close();
        return predictResult;
    }

    /** {@inheritDoc} */
    @Override
    protected Options getOptions() {
        return BertArguments.getOptions();
    }

    /** {@inheritDoc} */
    @Override
    protected BertArguments parseArguments(CommandLine cmd) {
        return new BertArguments(cmd);
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
