import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import * as dotenv from 'dotenv';
dotenv.config();


async function testChromaIngest() {

  console.log("⚡ Starting Vector Ingestion into ChromaDB...\n");

  // 1. Create sample documents
  const rawDocs = [
    new Document({
      pageContent: "LangChain is a framework designed to simplify the creation of applications using large language models (LLMs).",
      metadata: { source: "intro.txt" },
    }),
    new Document({
      pageContent: "ChromaDB is an open-source vector database designed for AI application embeddings and fast similarity retrieval.",
      metadata: { source: "chroma_docs.txt" },
    }),
  ];

  // 2. Chunk documents
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 150,
    chunkOverlap: 20,
  });
  const splitDocs = await splitter.splitDocuments(rawDocs);
  console.log(`✂️ Created ${splitDocs.length} chunks.`);

  // 3. Initialize Gemini Embeddings
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: "text-embedding-004",
  });

  // 4. Ingest into ChromaDB
  console.log("💾 Embedding and saving chunks to ChromaDB...");
  const vectorStore = await Chroma.fromDocuments(splitDocs, embeddings, {
    collectionName: "rag_demo_collection",
    url: "http://localhost:8000", // Ensure local Docker or Chroma server is running
  });

  console.log("✅ Successfully stored chunks in ChromaDB!\n");

  // 5. Test search query
  console.log("🔍 Running test similarity search for: 'What is ChromaDB?'");
  const searchResults = await vectorStore.similaritySearch("What is ChromaDB?", 1);

  console.log("\n--- TOP MATCH RETRIEVED FROM CHROMADB ---");
  console.log("Content:", searchResults[0].pageContent);
  console.log("Metadata:", searchResults[0].metadata);
}

testChromaIngest().catch((err) => {
  console.error("❌ Ingestion Error:", err);
});