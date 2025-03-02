package ai.djl.examples.inference.nlp;

import java.util.ArrayList;
import java.util.List;

import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Pair;

/**
 * Owlv2ImageTextTranslator
 * takes in image with class labels, and returns detected objects as float[] where
 * first item in array is the class number, next numbers are center_x, center_x, width, height
 * normalized to the original image width in the range [0, 1] 
 *
 * All functionality is derived from reading and exploring huggingface code: 
 * https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlv2/modeling_owlv2.py#L112
 */
public class Owlv2ImageTextTranslator 
	implements NoBatchifyTranslator<Pair<Image, List<String>>, List<float[]>>{

	// Confidence threshold for detection an object.
	private float detectionThreshold;
	private final int INPUT_IMG_WIDTH;
	private final int INPUT_IMG_HEIGHT;

	private Owlv2TextPreProcessor textPreProcessor;
	private Owlv2ImagePreProcessor imagePreProcessor;

	public Owlv2ImageTextTranslator() {
		detectionThreshold = 0.3f;
		INPUT_IMG_WIDTH = 960;
		INPUT_IMG_HEIGHT = 960;
		this.imagePreProcessor = new Owlv2ImagePreProcessor();
		this.textPreProcessor = new Owlv2TextPreProcessor();
	}

	@Override
	/**
	 * Given image and list of queries, return (inputIds, pixelValues, attentions)
	 */
	public NDList processInput(
			TranslatorContext ctx, Pair<Image, List<String>> input) throws Exception {
		
		List<String> queries = input.getValue();
		List<NDArray> inputIds = new ArrayList<>();
		List<NDArray> attentions = new ArrayList<>();

		for (int i = 0; i < queries.size(); i++) {
			NDList processedQuery = textPreProcessor.processInput(ctx, queries.get(i));
			inputIds.add(processedQuery.get(0));
			attentions.add(processedQuery.get(1));
		}

		inputIds = padMaxLength(ctx.getNDManager(), inputIds);
		attentions = padMaxLength(ctx.getNDManager(), attentions);

		NDArray allInputIds = stackArrays(inputIds);
		System.out.println("all input ids: " + allInputIds.toString());

		NDArray allAttentions = stackArrays(attentions);
		System.out.println("all attention masks: " + allAttentions.toString());

		NDArray pixelValues = imagePreProcessor.processInput(ctx, input.getKey()).get(0);
		System.out.println("pixel values: " + pixelValues.toString());
		return new NDList(allInputIds, pixelValues, allAttentions);
	}

	@Override
	public List<float[]> processOutput(
			TranslatorContext ctx, NDList list) throws Exception {
		NDArray predBoxes = list.get(2).squeeze();
		NDArray logits = list.get(0).squeeze();
		NDArray probabilities = Activation.sigmoid(logits);
		long numClasses = probabilities.getShape().getLastDimension();

		List<float[]> detectedObjects = new ArrayList<>();
		// One column of probabilities for each class.
		// Slice each column and filter where probability exceeds threshold
		for (int classIndex = 0; classIndex < numClasses; classIndex++) {
			NDIndex classProbabilitiesIndex = new NDIndex(":, %d".formatted(classIndex));
			float[] classProbs = probabilities
				.get(classProbabilitiesIndex)
				.toFloatArray();
			System.out.println(
					"probabilities for class %d: %f, %f, %f ... "
						.formatted(classIndex, classProbs[0], classProbs[1], classProbs[2]));
			for (int prediction = 0; prediction < classProbs.length; prediction++) {
				float probability = classProbs[prediction];
				if (probability >= detectionThreshold) {
					// Box is centerX, centerY, width, height normalized to image size in [0, 1]
					float[] predBox = predBoxes.get(prediction).toFloatArray();
					detectedObjects.add(new float[] {classIndex, predBox[0], predBox[1], predBox[2], predBox[3]});
					System.out.println(
							"detected object of class %d with probability %f"
								.formatted(classIndex, probability));
				}
			}
		}
		System.out.println(
				"detected %d objects".formatted(detectedObjects.size()));
		return detectedObjects;
	}

	/**
	 * Concatenate list of 1d arrays into an IxJ array where I is number of arrays, and J is the length of the arrays
	 * All arrays must be the same length.
	 */
	public static NDArray stackArrays(List<NDArray> arrays) {
		NDArray array = arrays.get(0);
		long arrayLength = array.size();
		NDArray concatenated = array.reshape(new Shape(1, arrayLength));
		if (arrays.size() > 1) {
			for (int i = 1; i < arrays.size(); i++) {
				array = arrays.get(i);
				array = array.reshape(new Shape(1, arrayLength));
				concatenated = concatenated.concat(array);
			}
		}
		return concatenated;
	}

	/**
	 * Pad a list of 1D NDArrays to the max length of the arrays with 0's
	 * and return the answer as a list of NDArrays
	 */
	public static List<NDArray> padMaxLength(NDManager manager, List<NDArray> arrays)  {
		long maxLength = -1;
		for (NDArray array : arrays) {
			if (array.size() > maxLength)
				maxLength = array.size();
		}
		System.out.println("max array length: " + maxLength);
		List<NDArray> results = new ArrayList<>();
		for (NDArray array : arrays) {
			if (array.size() < maxLength) {
				System.out.println("array is not less than max length");
				// Create a new array padded with zeros
				NDArray newArray = manager.zeros(new Shape(maxLength));
				// Copy the data from the old array to the new array
				NDIndex index = new NDIndex("0:%d".formatted(array.size()));
				newArray.set(index, array);
				System.out.println(
						"padded array with %d zeros to form new array: %s"
							.formatted(maxLength - array.size(), newArray.toString()));
				results.add(newArray);
			} else {
				System.out.println("array is not less than max length");
				results.add(array);
			}
		}
		return results;
	}

	public float getDetectionThreshold() {
		return detectionThreshold;
	}

	public void setDetectionThreshold(float detectionThreshold) {
		this.detectionThreshold = detectionThreshold;
	}

	public int getINPUT_IMG_WIDTH() {
		return INPUT_IMG_WIDTH;
	}

	public int getINPUT_IMG_HEIGHT() {
		return INPUT_IMG_HEIGHT;
	}

	public Owlv2TextPreProcessor getTextPreProcessor() {
		return textPreProcessor;
	}

	public void setTextPreProcessor(Owlv2TextPreProcessor textPreProcessor) {
		this.textPreProcessor = textPreProcessor;
	}

	public Owlv2ImagePreProcessor getImagePreProcessor() {
		return imagePreProcessor;
	}

	public void setImagePreProcessor(Owlv2ImagePreProcessor imagePreProcessor) {
		this.imagePreProcessor = imagePreProcessor;
	}

}
