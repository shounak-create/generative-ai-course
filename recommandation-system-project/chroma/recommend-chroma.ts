import { GoogleGenAI } from '@google/genai';
import { ChromaClient } from 'chromadb';

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error('❌ Error: GEMINI_API_KEY environment variable is missing.');
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey });
const chroma = new ChromaClient({ path: 'http://localhost:8000' });

async function getQueryEmbedding(queryText: string): Promise<number[]> {
  const response = await ai.models.embedContent({
    model: 'gemini-embedding-001',
    contents: queryText,
    config: { taskType: 'RETRIEVAL_QUERY' },
  });
  const values = response.embedding?.values ?? response.embeddings?.[0]?.values;
  if (!values) throw new Error('Failed to retrieve query embedding.');
  return values;
}

async function recommend(userQuery: string, topK = 3) {
  console.log(`\n🔎 User Query: "${userQuery}"`);

  const collection = await chroma.getCollection({ name: 'movies_collection' });
  const queryVector = await getQueryEmbedding(userQuery);

  // Perform vector search in ChromaDB
  const results = await collection.query({
    queryEmbeddings: [queryVector],
    nResults: topK,
  });

  console.log(`🎯 Top ${topK} ChromaDB Recommendations:`);
  console.log('──────────────────────────────────────────────────');

  let contextText = '';

  results.ids[0].forEach((id, index) => {
    const metadata = results.metadatas[0][index] as any;
    const distance = results.distances ? results.distances[0][index] : 'N/A';

    console.log(`${index + 1}. ${metadata.title} [Distance: ${distance}]`);
    console.log(`   Genres: ${metadata.genres}`);
    console.log(`   Plot:   ${metadata.overview}\n`);

    // Build context string for injection
    contextText += `Movie ${index + 1}: ${metadata.title}\nGenres: ${metadata.genres}\nOverview: ${metadata.overview}\n\n`;
  });

  // -------------------------------------------------------------
  // CONTEXT INJECTION (RAG Step)
  // Inject the retrieved ChromaDB context into the LLM prompt
  // -------------------------------------------------------------
  console.log('🤖 Generating AI Recommendation using injected context...');
  console.log('──────────────────────────────────────────────────');

  const prompt = `
You are an expert movie recommendation assistant.
A user is searching for: "${userQuery}"

Here is the context retrieved from our movie vector database:
${contextText}

Based strictly on the provided movie context above, provide a helpful and enthusiastic recommendation explaining why these movies fit the user's request.
  `;

  const aiResponse = await ai.models.generateContent({
    model: 'gemini-3.5-flash',
    contents: prompt,
  });

  console.log(aiResponse.text);
}

const searchQuery = process.argv.slice(2).join(' ') || 'space exploration and black holes';
recommend(searchQuery).catch(console.error);