@file:Suppress("UNCHECKED_CAST")

import ai.djl.Model
import ai.djl.modality.Input
import ai.djl.modality.Output
import ai.djl.modality.cv.Image
import ai.djl.modality.cv.output.CategoryMask
import ai.djl.modality.cv.output.Joints
import ai.djl.modality.cv.translator.*
import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.Vocabulary
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding
import ai.djl.nn.Block
import ai.djl.repository.zoo.Criteria
import ai.djl.translate.*
import java.nio.file.Path

inline fun <reified IN, reified OUT> BigGANTranslatorFactory.newInstance(model: Model, arguments: Map<String, *>): Translator<IN, OUT> {
    require(isSupported(IN::class.java, OUT::class.java)) { "Unsupported input/output types." }
    val truncation = ArgumentsUtil.floatValue(arguments, "truncation", 0.5f)
    return BigGANTranslator(truncation) as Translator<IN, OUT>
}

inline operator fun <reified IN, reified OUT> ExpansionTranslatorFactory<*, *>.invoke(model: Model, arguments: Map<String, *>): Translator<IN, OUT> =
    newInstance(IN::class.java, OUT::class.java, model, arguments)

inline operator fun <reified IN, reified OUT> SemanticSegmentationTranslatorFactory.invoke(arguments: Map<String, *>): Translator<IN, OUT> {
    require(IN::class == Image::class && OUT::class == CategoryMask::class) { "Unsupported input/output types." }
    return SemanticSegmentationTranslator.builder(arguments).build() as Translator<IN, OUT>
}

inline operator fun <reified IN, reified OUT> SimplePoseTranslatorFactory.invoke(arguments: Map<String, *>): Translator<IN, OUT> {
    val translator = simplePoseTranslator(arguments) {}
    return when {
        IN::class == Image::class && OUT::class == Joints::class -> translator as Translator<IN, OUT>
        IN::class == Input::class && OUT::class == Output::class -> ImageServingTranslator(translator) as Translator<IN, OUT>
        else -> throw java.lang.IllegalArgumentException("Unsupported input/output types.")
    }
}

inline fun simplePoseTranslator(arguments: Map<String, *>? = null, init: SimplePoseTranslatorDSL.() -> Unit): SimplePoseTranslator =
    SimplePoseTranslatorDSL(arguments).apply(init)()

class SimplePoseTranslatorDSL(arguments: Map<String, *>? = null) {
    val builder = arguments?.let(SimplePoseTranslator::builder) ?: SimplePoseTranslator.builder()
    var flag: Image.Flag
        @Deprecated(message = "Write only property", level = DeprecationLevel.HIDDEN) get() = TODO()
        set(value) {; builder.optFlag(value); }
    var batchifier: Batchifier
        @Deprecated(message = "Write only property", level = DeprecationLevel.HIDDEN) get() = TODO()
        set(value) {; builder.optBatchifier(value); }
    var threshold: Float
        @Deprecated(message = "Write only property", level = DeprecationLevel.HIDDEN) get() = TODO()
        set(value) {;builder.optThreshold(value); }
    var pipeline: Pipeline
        @Deprecated(message = "Write only property", level = DeprecationLevel.HIDDEN) get() = TODO()
        set(value) {;builder.setPipeline(value); }

    operator fun invoke(): SimplePoseTranslator = builder.build()
}

inline operator fun <reified IN, reified OUT> StyleTransferTranslatorFactory.invoke(): Translator<IN, OUT> {
    require(isSupported(IN::class.java, OUT::class.java)) { "Unsupported input/output types." }
    return StyleTransferTranslator() as Translator<IN, OUT>
}

inline fun defaultVocabulary(init: DefaultVocabularyDSL.() -> Unit): DefaultVocabulary =
    DefaultVocabularyDSL().apply(init)()

class DefaultVocabularyDSL {
    val builder = DefaultVocabulary.builder()
    fun add(list: List<String>) {; builder.add(list); }
    var maxTokens: Int
        @Deprecated(message = "Write only property", level = DeprecationLevel.HIDDEN) get() = TODO()
        set(value) {;builder.optMaxTokens(value); }

    fun unknownToken() {; builder.optUnknownToken(); }
    operator fun invoke(): DefaultVocabulary = builder.build()
}

inline fun trainableWordEmbedding(init: TrainableWordEmbeddingDSL.() -> Unit): TrainableWordEmbedding =
    TrainableWordEmbeddingDSL().apply(init)()

class TrainableWordEmbeddingDSL {
    val builder = TrainableWordEmbedding.builder()
    var vocabulary: Vocabulary
        @Deprecated(message = "Write only property", level = DeprecationLevel.HIDDEN) get() = TODO()
        set(value) {; builder.setVocabulary(value); }
    var useDefault: Boolean
        @Deprecated(message = "Write only property", level = DeprecationLevel.HIDDEN) get() = TODO()
        set(value) {;builder.optUseDefault(value); }

    operator fun invoke(): TrainableWordEmbedding = builder.build()
}

inline fun <reified IN, reified OUT> criteria(init: CriteriaDSL<IN, OUT>.() -> Unit): Criteria<IN, OUT> =
    CriteriaDSL<IN, OUT>().apply(init).build<IN, OUT>()

inline operator fun <reified IN, reified OUT> Criteria<IN, OUT>.invoke(init: CriteriaDSL<IN, OUT>.() -> Unit): Criteria<IN, OUT> =
    CriteriaDSL(toBuilder()).apply(init).build()

class CriteriaDSL<IN, OUT>(val builder: Criteria.Builder<IN, OUT> = Criteria.builder() as Criteria.Builder<IN, OUT>) {
    var block: Block
        @Deprecated(message = "Write only property", level = DeprecationLevel.HIDDEN) get() = TODO()
        set(value) {;builder.optBlock(value); }
    var modelPath: Path
        @Deprecated(message = "Write only property", level = DeprecationLevel.HIDDEN) get() = TODO()
        set(value) {;builder.optModelPath(value); }
    var modelUrls: String
        @Deprecated(message = "Write only property", level = DeprecationLevel.HIDDEN) get() = TODO()
        set(value) {;builder.optModelUrls(value); }
    var modelName: String
        @Deprecated(message = "Write only property", level = DeprecationLevel.HIDDEN) get() = TODO()
        set(value) {;builder.optModelName(value); }
    var engine: String
        @Deprecated(message = "Write only property", level = DeprecationLevel.HIDDEN) get() = TODO()
        set(value) {;builder.optEngine(value); }
    var translatorFactory: TranslatorFactory
        @Deprecated(message = "Write only property", level = DeprecationLevel.HIDDEN) get() = TODO()
        set(value) {;builder.optTranslatorFactory(value); }

    fun arguments(vararg pairs: Pair<String, String>) {
        for ((k, v) in pairs)
            builder.optArgument(k, v)
    }

    fun options(vararg pairs: Pair<String, String>) {
        for ((k, v) in pairs)
            builder.optOption(k, v)
    }

    @JvmName("build_IN_OUT")
    inline fun <reified IN, reified OUT> build(): Criteria<IN, OUT> = builder.setTypes(IN::class.java, OUT::class.java).build()
    fun build(): Criteria<IN, OUT> = builder.build()
}

fun main() {
    //    Criteria.builder()
    //            .setTypes(NDList::class.java, NDList::class.java)
    //            .optModelPath(path)
    //            .optArgument("blockFactory", "ai.djl.nn.OnesBlockFactory")
    //            .optArgument("block_shapes", "(1)s,(1)d,(1)u,(1)b,(1)i,(1)l,(1)B,(1)f,(1)")
    //            .optArgument("block_names", "1,2,3,4,5,6,7,8,9")
    //            .optOption("hasParameter", "false")
    //            .build();
}