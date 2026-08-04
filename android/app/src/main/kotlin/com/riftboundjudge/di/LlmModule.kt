package com.riftboundjudge.di

import com.riftboundjudge.data.llm.LiteRtLmRunner
import com.riftboundjudge.domain.LlmRunner
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
abstract class LlmModule {

    @Binds
    @Singleton
    abstract fun bindLlmRunner(impl: LiteRtLmRunner): LlmRunner
}