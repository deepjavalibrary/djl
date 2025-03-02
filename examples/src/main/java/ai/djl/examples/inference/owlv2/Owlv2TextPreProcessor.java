package test_djl.owl_v2;

import java.io.IOException;
import java.nio.file.Paths;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.PreProcessor;
import ai.djl.translate.TranslatorContext;

/**
 * The settings for the preprocessor are taken from the HuggingFace owl-vit AutoProcessor:
 * https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/image_processing_owlvit.py
 *
 * Assumes you have traced the CLIP model using the following code:
 * from transformers import CLIPModel, CLIPProcessor
 * import torch
 * from PIL import Image
 * import requests
 * model_name = "openai/clip-vit-base-patch32"
 * 
 * model = CLIPModel.from_pretrained(model_name, torchscript=True, return_dict=False)
 * processor = CLIPProcessor.from_pretrained(model_name)
 * 
 * test_text = "this is a cat"
 * text_inputs = processor(text=test_text, return_tensors="pt")
 * url = "http://images.cocodataset.org/val2017/000000039769.jpg"
 * image = Image.open(requests.get(url, stream=True).raw)
 * image_inputs = processor(images=image, return_tensors="pt")
 * converted = torch.jit.trace_module(model,  {'get_text_features': [text_inputs['input_ids'], text_inputs['attention_mask']],
 *                                             'get_image_features': [image_inputs['pixel_values']]})
 * torch.jit.save(converted, "clip-vit-base-patch32.pt")
 */
public class Owlv2TextPreProcessor implements PreProcessor<String> {

	// OWL-ViT uses CLIP tokenizer, see line 73: https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/processing_owlvit.py
	private final HuggingFaceTokenizer tokenizer; 

	public Owlv2TextPreProcessor() {
		try {
			tokenizer = HuggingFaceTokenizer.newInstance(Paths.get("src/main/resources/clip-vit-base-patch32"));
		} catch (IOException err) {
			throw new RuntimeException("could not open tokenizer: " + err);
		}
	}

	@Override
	public NDList processInput(TranslatorContext ctx, String input) throws Exception {
		Encoding encoding = tokenizer.encode(input);
		NDArray inputIds = ctx.getNDManager().create(encoding.getIds()).expandDims(0);
		NDArray attention = ctx.getNDManager().create(encoding.getAttentionMask()).expandDims(0);
		NDArray placeholder = ctx.getNDManager().create("");
		placeholder.setName("module_method:get_text_features");
		System.out.println(
				"query: %s, inputIds: %s, attention mask: %s"
					.formatted(input, inputIds, attention));
		return new NDList(inputIds, attention, placeholder);
	}

}
