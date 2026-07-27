import { GoogleGenAI } from '@google/genai';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error('❌ Error: GEMINI_API_KEY environment variable is missing.');
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey });

interface MovieWithVector {
  id: string;
  title: string;
  genres: string[];
  overview: string;
  vector: number[];
}

// 📐 Math: Calculate Cosine Similarity between two 768-dim vectors
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// 🔤 Embed Search Query using RETRIEVAL_QUERY
async function getQueryEmbedding(queryText: string): Promise<number[]> {
  const response = await ai.models.embedContent({
    model: 'gemini-embedding-001',
    contents: queryText,
    config: {
      taskType: 'RETRIEVAL_QUERY', // Query taskType for searching
    },
  });

  const values = response.embedding?.values ?? response.embeddings?.[0]?.values;
  if (!values) {
    throw new Error('Failed to retrieve query embedding values.');
  }

  return values;
}

async function recommend(userQuery: string, topK = 3) {
  const dbPath = path.join(process.cwd(), 'movies_with_vectors.json');

  // 1. Read vector database from JSON
  const fileContent = await fs.readFile(dbPath, 'utf-8');
  const movies: MovieWithVector[] = JSON.parse(fileContent);

  console.log(`\n🔎 User Query: "${userQuery}"`);
  console.log('⏳ Embedding query and computing similarities...\n');

  // 2. Convert user search string into a vector
  const queryVector = await getQueryEmbedding(userQuery);

  // 3. Compare user query vector against every movie vector
  const scoredMovies = movies.map((movie) => {
    const similarity = cosineSimilarity(queryVector, movie.vector);
    return {
      title: movie.title,
      genres: movie.genres,
      overview: movie.overview,
      score: similarity,
    };
  });

  // 4. Sort descending by score
  scoredMovies.sort((a, b) => b.score - a.score);

  // 5. Display top results
  console.log(`🎯 Top ${topK} Recommendations:`);
  console.log('──────────────────────────────────────────────────');

  scoredMovies.slice(0, topK).forEach((movie, index) => {
    const percentage = (movie.score * 100).toFixed(2);
    console.log(`${index + 1}. ${movie.title} [Match: ${percentage}%]`);
    console.log(`   Genres: ${movie.genres.join(', ')}`);
    console.log(`   Plot:   ${movie.overview}\n`);
  });
}

// Get search query from command line args or use default
const searchQuery = process.argv.slice(2).join(' ') || 'mind bending sci-fi space travel';

recommend(searchQuery).catch(console.error);