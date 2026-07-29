import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";

async function testTextSplitter() {
  console.log("✂️ Testing Text Chunking Pipeline...\n");

  // Mock raw document representing a single large page or loaded document
  const sampleDocument = new Document({
    pageContent: `
    Retrieval-Augmented Generation (RAG) is a technique that grants Large Language Models (LLMs) access to external knowledge bases. 
    By querying a database for relevant context before asking the model a question, RAG prevents hallucinations and provides accurate, domain-specific answers.

    Vector databases like ChromaDB and LanceDB store data as high-dimensional numerical vectors, known as embeddings. 
    When a user asks a question, the pipeline converts the query into an embedding and calculates similarity metrics (such as cosine similarity) to retrieve the top matching text chunks.

    LangChain serves as the orchestration layer, connecting document loaders, text splitters, vector stores, and prompt templates into an automated chain.
    `,
    metadata: { source: "rag_guide.md", author: "Shounak" },
  });

  // Configure chunking parameters
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,   // Target character count per chunk
    chunkOverlap: 40, // Character overlap between adjacent chunks to preserve boundary context
  });

  // Split document into chunks
  const splitDocs = await textSplitter.splitDocuments([sampleDocument]);

  console.log(`✅ Original Document Split into ${splitDocs.length} Chunks.\n`);
  console.log("-----------------------------------------");

  splitDocs.forEach((chunk, index) => {
    console.log(`--- CHUNK ${index + 1} (${chunk.pageContent.length} chars) ---`);
    console.log(chunk.pageContent.trim());
    console.log("Metadata:", chunk.metadata);
    console.log();
  });
}

testTextSplitter().catch(console.error);