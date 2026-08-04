package com.riftboundjudge.ui.library

import com.riftboundjudge.domain.Document

sealed interface LibraryUiState {
    data object Loading : LibraryUiState
    data object Empty : LibraryUiState
    data class Loaded(val docs: List<Document>) : LibraryUiState
}