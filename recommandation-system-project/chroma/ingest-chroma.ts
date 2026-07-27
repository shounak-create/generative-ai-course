import { GoogleGenAI } from '@google/genai';
import { ChromaClient } from 'chromadb';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error('❌ Error: GEMINI_API_KEY environment variable is missing.');
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey });
const chroma = new ChromaClient({ path: 'http://localhost:8000' });

interface Movie {
  id: string;
  title: string;
  genres: string[];
  overview: string;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function getEmbedding(text: string): Promise<number[]> {
  const response = await ai.models.embedContent({
    model: 'gemini-embedding-001',
    contents: text,
    config: { taskType: 'RETRIEVAL_DOCUMENT' },
  });
  const values = response.embedding?.values ?? response.embeddings?.[0]?.values;
  if (!values) throw new Error('No embedding returned.');
  return values;
}

async function runChromaIngestion() {
  const rawDataPath = path.join(process.cwd(), 'movies.json');
  const rawMovies: Movie[] = JSON.parse(await fs.readFile(rawDataPath, 'utf-8'));

  // Create or get collection in ChromaDB
  const collection = await chroma.getOrCreateCollection({ name: 'movies_collection' });

  console.log(`⏳ Embedding and inserting ${rawMovies.length} movies into ChromaDB...`);

  for (const movie of rawMovies) {
    const richText = `Title: ${movie.title}. Genres: ${movie.genres.join(', ')}. Plot: ${movie.overview}`;
    console.log(`  └─ Embedding "${movie.title}"...`);
    
    const vector = await getEmbedding(richText);

    // Insert document into ChromaDB
    await collection.add({
      ids: [movie.id],
      embeddings: [vector],
      metadatas: [{ title: movie.title, genres: movie.genres.join(', '), overview: movie.overview }],
      documents: [richText],
    });

    await sleep(300);
  }

  console.log('✨ Success! ChromaDB collection "movies_collection" updated.');
}

runChromaIngestion().catch(console.error);