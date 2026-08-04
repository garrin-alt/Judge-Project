package com.riftboundjudge.domain

interface EmbeddingService {
    suspend fun embed(text: String): FloatArray
}