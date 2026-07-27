import { GoogleGenAI } from '@google/genai';
import * as readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error('❌ Error: GEMINI_API_KEY environment variable is missing.');
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey });

const documents = [
  'Node.js uses an event-driven, non-blocking I/O model making it lightweight and efficient.',
  'TypeScript adds static type definitions to JavaScript to improve code quality and maintainability.',
  'Italian Carbonara uses egg yolks, guanciale, Pecorino Romano cheese, and black pepper.',
  'Git is a distributed version control system tracking changes in source code during software development.',
  'The MERN stack consists of MongoDB, Express.js, React, and Node.js for full-stack development.',
  'High-protein diets with chicken breast and eggs help with muscle repair and recovery.',
];

// 📐 Math Helper: Measures the angle between two vectors (-1 to 1)
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// 🔤 Embedder Helper: Converts text into a 768-dimensional vector
async function getEmbedding(text: string, taskType: 'RETRIEVAL_DOCUMENT' | 'RETRIEVAL_QUERY'): Promise<number[]> {
  const response = await ai.models.embedContent({
    model: 'gemini-embedding-001',
    contents: text,
    config: {
      taskType: taskType, // Optimizes vector geometry for search
    },
  });

  // Extract vector values safely across SDK versions
  const values = response.embedding?.values ?? response.embeddings?.[0]?.values;

  if (!values) {
    throw new Error('Failed to retrieve embedding values from API response.');
  }

  return values;
}

async function main() {
  console.log('⏳ Generating embeddings for knowledge base documents...');

  // 1. Pre-embed all documents with RETRIEVAL_DOCUMENT taskType
  const documentDatabase = await Promise.all(
    documents.map(async (doc) => ({
      text: doc,
      vector: await getEmbedding(doc, 'RETRIEVAL_DOCUMENT'),
    }))
  );

  console.log(`✅ Success! Indexed ${documentDatabase.length} documents into vector space.\n`);

  const rl = readline.createInterface({ input, output });

  try {
    while (true) {
      const query = await rl.question('🔎 Enter search query (or "exit"): ');
      if (query.trim().toLowerCase() === 'exit') break;
      if (!query.trim()) continue;

      // 2. Embed the user search query with RETRIEVAL_QUERY taskType
      const queryVector = await getEmbedding(query, 'RETRIEVAL_QUERY');

      // 3. Compute cosine similarity score for every document
      const rankedResults = documentDatabase
        .map((doc) => ({
          text: doc.text,
          score: cosineSimilarity(queryVector, doc.vector),
        }))
        .sort((a, b) => b.score - a.score);

      // 4. Output the top 3 closest matches
      console.log('\n--- Top Matches ---');
      rankedResults.slice(0, 3).forEach((result, rank) => {
        const percentage = (result.score * 100).toFixed(1);
        console.log(`${rank + 1}. [Match: ${percentage}%] ${result.text}`);
      });
      console.log('-------------------\n');
    }
  } finally {
    rl.close();
  }
}

main().catch(console.error);