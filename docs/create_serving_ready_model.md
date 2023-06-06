# Create a serving ready model

To deploy a machine learning model for inference usually involve more than the model artifacts. In
most cases, developer has to handle pre-process, post-process and batching for inference. DJL
introduces [Translator](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/translate/Translator.html)
interface to handle most of the boilerplate code and allows developer focus on their model logic.

There are many state-of-the-art models published publicly. Due to the complicity nature of the data
processing, developers still need to dig into examples, original training scripts or even contact
the original author to figure out how to implement the data processing. DJL provides two ways to
address this gap.

## ModelZoo
Models in the DJL ModelZoo are ready to use. End user don't need to worry about data processing.
DJL's ModelZoo allows you easily organize different type of models and their versions.
However, creating a custom model zoo isn't straightforward. We are still working on the tooling to
make it easy for model authors to create their own model zoo.

## Bundle your data processing scripts together with model artifacts
DJL allows model author to create a [ServingTranslator](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/translate/ServingTranslator.html)
class together with the model artifacts. DJL will load the bundled `ServingTranslator` and use
this class to conduct the data processing.

### Step 1: Create a ServingTranslator class
Create a java class that implements [ServingTranslator](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/translate/ServingTranslator.html)
interface. See: [MyTranslator](https://github.com/deepjavalibrary/djl/blob/master/integration/src/test/translator/MyTranslator.java) as an example.

### Step 2: Create a `libs` folder in your model directory
DJL will look into `libs` folder to search for Translator implementation.

### Step 3: Copy Translator into `libs` folder
DJL can load Translator from the following source:

- from jar files directly locate in `libs` folder
- from compiled java .class file in `libs/classes` folder
- DJL can compile .java files in `libs/classes` folder at runtime and load compiled class

## Configure data processing based on standard Translator
DJL provides several built-in Translator for well-know ML applications, such as `Image Classification`
and `Object Detection`. You can customize those built-in Translators' behavior by providing
configuration parameters.

There are two ways to supply configurations to the `Translator`:

- Add a `serving.properties` file in the model's folder

    Here is an example:

```config
# serving.properties can be used to define model's metadata, all the arguments will be
# passed to TranslatorFactory to create proper Translator

# defines model's application
application=nlp/question_answer

# defines the model's engine, can be overrid by Criteria.optEngine()
engine=PyTorch

# defines TranslatorFactory, can be overrid by Criteria.optTranslator() or Criteria.optTranslatorFactory()
translatorFactory=ai.djl.modality.cv.translator.ImageClassificationTranslatorFactory

# Add Translator specific arguments here to customize pre-processing and post-processing
# specify image size to be cropped
width=224
height=224
# specify the input image should be treated as grayscale image
flag=GRAYSCALE
# specify if apply softmax for post-processing
softmax=true
```

- Pass arguments in Criteria:

    You can customize Translator's behavior with Criteria, for example:

```java
Criteria<Image, Classifications> criteria = Criteria.builder()
        .setTypes(Image.class, Classifications.class) // defines input and output data type
        .optApplication(Application.CV.IMAGE_CLASSIFICATION) // spcific model's application
        .optModelUrls("file:///var/models/my_resnet50") // search models in specified path
        .optArgument("width", 224)
        .optArgument("height", 224)
        .optArgument("height", 224)
        .optArgument("flag", "GRAYSCALE")
        .optArgument("softmax", true)
        .build();
```
