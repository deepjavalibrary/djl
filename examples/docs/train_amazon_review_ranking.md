# Train Amazon Review Ranking Dataset using DistilBert

In this example, you learn how to train the Amazon Review dataset.
This dataset includes 30k reviews from Amazon customers on different products.
We only use `review_body` and `star_rating` for data and label.

You can find the example source code in: [TrainAmazonReviewRanking.java](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/training/transferlearning/TrainAmazonReviewRanking.java).

## Setup guide

Follow [setup](../../docs/development/setup.md) to configure your development environment.

## Train the model

In this example, we used the [GluonNLP pretrained DistilBert](https://nlp.gluon.ai/model_zoo/bert/index.html) model followed by a simple MLP layer.
The input is the BERT formatted tokens and the output is the star rating.
We recommend using GPU for training since CPU training is slow with this dataset.

```bash
cd examples
./gradlew run -Dmain=ai.djl.examples.training.transferlearning.TrainAmazonReviewRanking --args="-e 2 -b 8 -g 1"
```

You can adjust the `maxTokenLength` variable (currently 64) to a larger value to achieve better accuracy.
