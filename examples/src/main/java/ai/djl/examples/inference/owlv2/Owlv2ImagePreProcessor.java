package ai.djl.examples.inference.nlp;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.PreProcessor;
import ai.djl.translate.TranslatorContext;

/**
 * All we need to do is resize the image and convert it to a tensor.
 * All the other preprocessing stuff such as normalization and scaling is handled by the
 * traced .pt model.
 *
 * https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/image_processing_owlvit.py
 */
public class Owlv2ImagePreProcessor implements PreProcessor<Image> {

	private final int INPUT_IMG_WIDTH = 960;
	private final int INPUT_IMG_HEIGHT = 960;
	private final float[] OPENAI_IMG_MEAN = { 0.48145466f, 0.4578275f, 0.40821073f };
	private final float[] OPENAI_IMG_STD = { 0.26862954f, 0.26130258f, 0.27577711f };
	private final float RESCALE_FACTOR = 1.0f / 255.0f;

	@Override
	public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
		NDArray imageND = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
		imageND = NDImageUtils.resize(
				imageND, INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT, Image.Interpolation.BICUBIC);
		imageND = NDImageUtils.toTensor(imageND);
		NDArray placeholder = ctx.getNDManager().create("");
		placeholder.setName("module_method:get_image_features");
		return new NDList(imageND.expandDims(0), placeholder);
	}

}
