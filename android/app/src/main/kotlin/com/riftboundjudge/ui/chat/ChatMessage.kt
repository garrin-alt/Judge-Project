package com.riftboundjudge.ui.chat

import com.riftboundjudge.domain.RetrievedChunk

enum class Role { User, Assistant }

data class ChatMessage(
    val id: Long,
    val role: Role,
    val text: String,
    val sources: List<RetrievedChunk> = emptyList(),
)