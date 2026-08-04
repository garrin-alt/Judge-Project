package com.riftboundjudge.data.repository

import com.riftboundjudge.data.Cosine
import com.riftboundjudge.data.db.ChunkDao
import com.riftboundjudge.data.embedding.toFloatArray
import com.riftboundjudge.di.DefaultDispatcher
import com.riftboundjudge.domain.RetrievedChunk
import com.riftboundjudge.domain.Retriever
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class RetrieverImpl @Inject constructor(
    private val chunkDao: ChunkDao,
    @DefaultDispatcher private val dispatcher: CoroutineDispatcher,
) : Retriever {

    override suspend fun topK(
        query: FloatArray,
        documentIds: List<String>?,
        k: Int,
    ): List<RetrievedChunk> = withContext(dispatcher) {
        // TODO: replace with ANN if chunk count grows beyond a few thousand
        val entities = if (documentIds != null) chunkDao.getForDocuments(documentIds) else chunkDao.getAll()
        entities
            .map { entity ->
                RetrievedChunk(
                    text = entity.text,
                    documentId = entity.documentId,
                    score = Cosine.similarity(query, entity.embedding.toFloatArray()),
                )
            }
            .sortedByDescending { it.score }
            .take(k)
    }
}