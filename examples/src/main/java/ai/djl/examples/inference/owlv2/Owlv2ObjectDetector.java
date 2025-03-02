package test_djl.owl_v2;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.util.Pair;

/**
 * OWLViTObjectDetector
 * https://huggingface.co/google/owlvit-base-patch32
 *
 * Assumes you have traced the OwlV2 model using the following python code:
 *
 * ```py
 * import numpy as np
 * from PIL import Image, ImageDraw
 * import requests
 * from transformers import Owlv2Processor, Owlv2ForObjectDetection
 * import torch
 * 
 * processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
 * model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", 
 *                                                  torchscript=True,
 *                                                  return_dict=False)
 * test_text = ["this is a dog", "cat's ear"]
 * text_inputs = processor(text=test_text, return_tensors="pt")
 * url = "http://images.cocodataset.org/val2017/000000039769.jpg"
 * image = Image.open(requests.get(url, stream=True).raw)
 * image = image.resize((960, 960))
 * image_inputs = processor(images=image, return_tensors="pt")
 * converted = torch.jit.trace_module(model, {"forward": (text_inputs.input_ids, image_inputs.pixel_values, text_inputs.attention_mask)})
 * converted.save("owlv2-base-patch16-ensemble-for-object-detection-traced-from-hf.pt")
 * ```
 */
public class Owlv2ObjectDetector {

	private ZooModel<NDList, NDList> model;
	private Translator<Pair<Image, List<String>>, List<float[]>> imageTextTranslator;
	private Predictor<Pair<Image, List<String>>, List<float[]>> predictor;

    public Owlv2ObjectDetector() throws ModelException, IOException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelPath(Paths.get("./src/main/resources/owlv2-base-patch16-ensemble-for-object-detection-traced-from-hf.pt"))
                        .optTranslator(new NoopTranslator())
                        .optEngine("PyTorch")
                        .optDevice(Device.cpu()) // torchscript model only support CPU
                        .build();
        model = criteria.loadModel();
		imageTextTranslator = new Owlv2ImageTextTranslator();
		predictor = model.newPredictor(imageTextTranslator);
    }

	/**
	 * Given image and list of possible labels, return detected objects as float[]
	 * where the item 1 of the array is the index of the label, and 2-5 are
	 * center x, center y, width, and height normalized to the size of the image
	 * in the range [0, 1]
	 */
	public List<float[]> predict(
			Pair<Image, List<String>> input) throws TranslateException{
		return predictor.predict(input);
	}
	
	
}
