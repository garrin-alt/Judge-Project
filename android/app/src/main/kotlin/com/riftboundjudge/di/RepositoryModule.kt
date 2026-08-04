package com.riftboundjudge.di

import com.riftboundjudge.data.repository.DocumentRepositoryImpl
import com.riftboundjudge.data.repository.RetrieverImpl
import com.riftboundjudge.domain.DocumentRepository
import com.riftboundjudge.domain.RagPipeline
import com.riftboundjudge.domain.RagPipelineImpl
import com.riftboundjudge.domain.Retriever
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
abstract class RepositoryModule {

    @Binds
    @Singleton
    abstract fun bindDocumentRepository(impl: DocumentRepositoryImpl): DocumentRepository

    @Binds
    @Singleton
    abstract fun bindRetriever(impl: RetrieverImpl): Retriever

    @Binds
    @Singleton
    abstract fun bindRagPipeline(impl: RagPipelineImpl): RagPipeline
}
