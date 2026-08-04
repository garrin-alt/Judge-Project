package com.riftboundjudge.data.embedding

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File

class BertTokenizerTest {

    companion object {
        private lateinit var tokenizer: BertTokenizer
        private lateinit var idToToken: Map<Int, String>

        @BeforeClass
        @JvmStatic
        fun setup() {
            // Load vocab from app/src/main/assets/embedding/vocab.txt. Relative to the
            // app module's project dir, which is the JVM test task's working directory
            // by default — portable across checkouts and CI runners, unlike an absolute path.
            val vocabFile = File("src/main/assets/embedding/vocab.txt")
            val vocabList = vocabFile.readLines().filter { it.isNotEmpty() }
            tokenizer = BertTokenizer(vocabList)
            idToToken = vocabList.withIndex().associate { (idx, token) -> idx to token }
        }
    }

    /**
     * Assert that [CLS] and [SEP] bracket the token sequence for a simple sentence.
     */
    @Test
    fun `CLS and SEP bracket token sequence`() {
        val (inputIds, _, _) = tokenizer.encode("hello world", maxLen = 128)

        // [CLS] is at index 101
        assertEquals("First token should be [CLS]", "[CLS]", idToToken[inputIds[0]])

        // [SEP] should be somewhere after the first token and before padding (0s)
        var foundSep = false
        for (i in 1 until inputIds.size) {
            if (inputIds[i] == 0) break // Reached padding
            if (idToToken[inputIds[i]] == "[SEP]") {
                foundSep = true
                break
            }
        }
        assertTrue("Should have [SEP] token", foundSep)
    }

    /**
     * Assert that a rule identifier like "702.12.b.2" tokenizes to more than one subword piece
     * (proving it isn't reduced to a single [UNK]) — the decoded token strings are NOT all [UNK].
     */
    @Test
    fun `rule identifier is not tokenized to all UNK`() {
        val (inputIds, _, _) = tokenizer.encode("702.12.b.2", maxLen = 128)

        // Collect all non-padding tokens (skip [CLS] and [SEP])
        val tokens = mutableListOf<String>()
        for (id in inputIds) {
            if (id == 0) break // Padding
            val token = idToToken[id] ?: "[UNKNOWN_ID]"
            if (token != "[CLS]" && token != "[SEP]") {
                tokens.add(token)
            }
        }

        // Should have at least one token (unlikely to be all [UNK])
        assertTrue("Should have tokens for rule identifier", tokens.isNotEmpty())

        // Not all tokens should be [UNK]
        val allUnk = tokens.all { it == "[UNK]" }
        assertFalse("Rule identifier should not be tokenized to all [UNK]", allUnk)
    }

    /**
     * Assert that a domain word like "Riftbound" or "showdown" is not a single [UNK] result.
     * WordPiece may split it into subwords, but it shouldn't be [UNK] for the whole word.
     */
    @Test
    fun `domain word riftbound is not all UNK`() {
        val (inputIds, _, _) = tokenizer.encode("riftbound", maxLen = 128)

        // Collect all non-padding tokens (skip [CLS] and [SEP])
        val tokens = mutableListOf<String>()
        for (id in inputIds) {
            if (id == 0) break // Padding
            val token = idToToken[id] ?: "[UNKNOWN_ID]"
            if (token != "[CLS]" && token != "[SEP]") {
                tokens.add(token)
            }
        }

        // Should have at least one token
        assertTrue("Should have tokens for riftbound", tokens.isNotEmpty())

        // Not all tokens should be [UNK]
        val allUnk = tokens.all { it == "[UNK]" }
        assertFalse("riftbound should not be tokenized to all [UNK]", allUnk)
    }

    /**
     * Assert that "showdown" is not a single [UNK].
     */
    @Test
    fun `domain word showdown is not all UNK`() {
        val (inputIds, _, _) = tokenizer.encode("showdown", maxLen = 128)

        // Collect all non-padding tokens (skip [CLS] and [SEP])
        val tokens = mutableListOf<String>()
        for (id in inputIds) {
            if (id == 0) break // Padding
            val token = idToToken[id] ?: "[UNKNOWN_ID]"
            if (token != "[CLS]" && token != "[SEP]") {
                tokens.add(token)
            }
        }

        // Should have at least one token
        assertTrue("Should have tokens for showdown", tokens.isNotEmpty())

        // Not all tokens should be [UNK]
        val allUnk = tokens.all { it == "[UNK]" }
        assertFalse("showdown should not be tokenized to all [UNK]", allUnk)
    }

    /**
     * Assert that output arrays are all exactly length 128 (maxLen), confirming padding.
     */
    @Test
    fun `output arrays are exactly maxLen length`() {
        val (inputIds, attentionMask, tokenTypeIds) = tokenizer.encode("hello", maxLen = 128)

        assertEquals("inputIds should be length 128", 128, inputIds.size)
        assertEquals("attentionMask should be length 128", 128, attentionMask.size)
        assertEquals("tokenTypeIds should be length 128", 128, tokenTypeIds.size)
    }

    /**
     * Assert that attention mask is 1 for non-padding tokens and 0 for padding.
     */
    @Test
    fun `attention mask correctly identifies padding`() {
        val (inputIds, attentionMask, _) = tokenizer.encode("hi", maxLen = 128)

        // First few should be 1 (non-padding)
        assertTrue("Early tokens should have attention mask 1", attentionMask[0] == 1)

        // Last tokens should be 0 (padding)
        assertTrue("Late tokens should have attention mask 0", attentionMask[127] == 0)

        // Verify consistency: mask[i] == 0 iff inputIds[i] == 0
        for (i in inputIds.indices) {
            val expectedMask = if (inputIds[i] == 0) 0 else 1
            assertEquals("Mask should match padding at index $i", expectedMask, attentionMask[i])
        }
    }
}
