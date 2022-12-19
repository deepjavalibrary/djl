# CLIP model example

[CLIP model](https://huggingface.co/openai/clip-vit-base-patch32) is open-sourced by OpenAI for text-image understanding.
It is widely used to get text and image feature and used for search domain.  User could use image to search image, use text to search image and even use text to search text with this model.

In this short demo, we will do an image to text comparison to find which text is close to the corresponding image.

The image we used is

![](http://images.cocodataset.org/val2017/000000039769.jpg)

And our input text:

```
"A photo of cats";
"A photo of dogs";
```

We expect cats text will win based on the image.

## Run the example

```
cd examples
./gradlew run -Dmain=ai.djl.examples.inference.clip.ImageTextComparison
```

output:

```
[INFO ] - A photo of cats Probability: 0.9970879546345841
[INFO ] - A photo of dogs Probability: 0.002912045365415886

```
