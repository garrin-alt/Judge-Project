package com.riftboundjudge.domain

data class Document(
    val id: String,
    val title: String,
    val uri: String,
    val createdAt: Long,
    val chunkCount: Int,
)
