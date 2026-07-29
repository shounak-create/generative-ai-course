import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { Document } from "@langchain/core/documents";
import * as dotenv from 'dotenv';
dotenv.config();

async function runRAGChain() {
  console.log("🚀 Initializing End-to-End RAG Pipeline...\n");

  // 1. Correct property is `modelName`, not `model`
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    modelName: "text-embedding-004", // MUST be modelName in @langchain/google-genai
  });

  // Re-connect to ChromaDB Vector Store
  const vectorStore = await Chroma.fromExistingCollection(embeddings, {
    collectionName: "rag_demo_collection",
    url: "http://localhost:8000",
  });

  // Convert VectorStore to Retriever (fetch top 2 relevant chunks)
  const retriever = vectorStore.asRetriever({ k: 2 });

  // 2. Initialize Gemini LLM (LLM uses `model`, embeddings use `modelName`)
  const model = new ChatGoogleGenerativeAI({
    model: "gemini-1.5-flash",
    apiKey: process.env.GEMINI_API_KEY,
    temperature: 0.2,
  });

  // 3. Define Prompt Template
  const promptTemplate = PromptTemplate.fromTemplate(`
You are a helpful assistant. Answer the user's question using ONLY the provided context below.
If the answer cannot be deduced from the context, respond with "I cannot answer based on the available information."

--- CONTEXT ---
{context}

--- QUESTION ---
{question}
`);

  // Helper function to format retrieved documents
  const formatDocs = (docs: Document[]) => {
    return docs.map((doc, idx) => `[Doc ${idx + 1}]: ${doc.pageContent}`).join("\n\n");
  };

  // 4. Construct LCEL Chain
  const ragChain = RunnableSequence.from([
    {
      context: retriever.pipe(formatDocs),
      question: new RunnablePassthrough(),
    },
    promptTemplate,
    model,
    new StringOutputParser(),
  ]);

  // 5. Execute Chain
  const query = "What is ChromaDB and what is it used for?";
  console.log(`🔍 Query: "${query}"\n`);
  console.log("⚡ Executing RAG Chain...");

  const response = await ragChain.invoke(query);

  console.log("\n--- FINAL RAG RESPONSE ---");
  console.log(response);
}

runRAGChain().catch((err) => {
  console.error("❌ RAG Pipeline Error:", err);
});