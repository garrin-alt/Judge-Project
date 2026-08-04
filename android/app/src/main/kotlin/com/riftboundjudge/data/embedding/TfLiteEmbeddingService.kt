package com.riftboundjudge.data.embedding

import com.riftboundjudge.domain.EmbeddingService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.sqrt

@Singleton
class TfLiteEmbeddingService @Inject constructor(
    private val assetLoader: AssetLoader
) : EmbeddingService {

    private val vocab: Map<String, Int> by lazy { assetLoader.loadVocab() }

    private val tokenizer: BertTokenizer by lazy {
        BertTokenizer(vocab.entries.sortedBy { it.value }.map { it.key })
    }

    private val interpreter: Interpreter by lazy {
        Interpreter(assetLoader.loadModel(), Interpreter.Options().setNumThreads(4))
    }

    override suspend fun embed(text: String): FloatArray = withContext(Dispatchers.Default) {
        val (inputIds, attentionMask, _) = tokenizer.encode(text, MAX_SEQ_LEN)
        val output = Array(1) { FloatArray(EMBEDDING_DIM) }

        interpreter.runForMultipleInputsOutputs(
            arrayOf(arrayOf(inputIds), arrayOf(attentionMask)),
            mapOf(0 to output)
        )
        l2Normalize(output[0])
    }

    private fun l2Normalize(vec: FloatArray): FloatArray {
        val norm = sqrt(vec.fold(0f) { acc, v -> acc + v * v })
        return FloatArray(vec.size) { vec[it] / norm.coerceAtLeast(1e-9f) }
    }

    companion object {
        private const val MAX_SEQ_LEN = 128
        private const val EMBEDDING_DIM = 384
    }
}