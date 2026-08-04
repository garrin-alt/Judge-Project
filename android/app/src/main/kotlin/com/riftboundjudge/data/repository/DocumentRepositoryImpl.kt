package com.riftboundjudge.data.repository

import android.net.Uri
import com.riftboundjudge.data.db.ChunkDao
import com.riftboundjudge.data.db.ChunkEntity
import com.riftboundjudge.data.db.DocumentDao
import com.riftboundjudge.data.db.DocumentEntity
import com.riftboundjudge.data.embedding.TfLiteEmbeddingService
import com.riftboundjudge.data.embedding.toByteArray
import com.riftboundjudge.data.pdf.Chunker
import com.riftboundjudge.data.pdf.PdfTextExtractor
import com.riftboundjudge.domain.Document
import com.riftboundjudge.domain.DocumentRepository
import com.riftboundjudge.domain.EmbeddingService
import com.riftboundjudge.domain.IngestProgress
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class DocumentRepositoryImpl @Inject constructor(
    private val pdf: PdfTextExtractor,
    private val chunker: Chunker,
    private val embed: EmbeddingService,
    private val docDao: DocumentDao,
    private val chunkDao: ChunkDao,
) : DocumentRepository {

    override fun observeAll(): Flow<List<Document>> =
        docDao.observeAll().map { entities -> entities.map { it.toDomain() } }

    override suspend fun delete(id: String) {
        val entity = docDao.getById(id) ?: return
        docDao.delete(entity)
    }

    override fun ingest(uriString: String, title: String): Flow<IngestProgress> = flow {
        emit(IngestProgress.Started)

        val text = pdf.extract(Uri.parse(uriString))
        emit(IngestProgress.Extracting)

        val chunks = chunker.chunk(text)
        emit(IngestProgress.Chunking(chunks.size))

        val docId = UUID.randomUUID().toString()
        val chunkEntities = ArrayList<ChunkEntity>(chunks.size)

        chunks.forEachIndexed { index, chunkText ->
            val vec = embed.embed(chunkText)
            chunkEntities.add(
                ChunkEntity(
                    documentId = docId,
                    ordinal = index,
                    text = chunkText,
                    embedding = vec.toByteArray(),
                )
            )
            emit(IngestProgress.Embedding(index + 1, chunks.size))
        }

        docDao.insert(
            DocumentEntity(
                id = docId,
                title = title,
                uri = uriString,
                createdAt = System.currentTimeMillis(),
                chunkCount = chunks.size,
            )
        )
        chunkDao.insertAll(chunkEntities)

        emit(IngestProgress.Done(docId))
    }.catch { e ->
        emit(IngestProgress.Error(e))
    }.flowOn(Dispatchers.IO)

    private fun DocumentEntity.toDomain() = Document(
        id = id,
        title = title,
        uri = uri,
        createdAt = createdAt,
        chunkCount = chunkCount,
    )
}
