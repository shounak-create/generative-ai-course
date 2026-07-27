import { GoogleGenAI } from '@google/genai';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error('❌ Error: GEMINI_API_KEY environment variable is missing.');
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey });

interface Movie {
  id: string;
  title: string;
  genres: string[];
  overview: string;
}

interface MovieWithVector extends Movie {
  vector: number[];
}

// 🔤 Convert text into a 768-dimensional vector
async function getEmbedding(text: string): Promise<number[]> {
  const response = await ai.models.embedContent({
    model: 'gemini-embedding-001',
    contents: text,
    config: {
      taskType: 'RETRIEVAL_DOCUMENT', // Document taskType for indexed content
    },
  });

  const values = response.embedding?.values ?? response.embeddings?.[0]?.values;
  if (!values) {
    throw new Error('Failed to retrieve embedding values.');
  }

  return values;
}

async function runIngestion() {
  const rawDataPath = path.join(process.cwd(), 'movies.json');
  const outputPath = path.join(process.cwd(), 'movies_with_vectors.json');

  console.log('📂 Reading raw movie catalog from movies.json...');
  const fileContent = await fs.readFile(rawDataPath, 'utf-8');
  const rawMovies: Movie[] = JSON.parse(fileContent);

  console.log(`⏳ Generating embeddings for ${rawMovies.length} movies...`);

  const moviesWithVectors: MovieWithVector[] = [];

  for (const movie of rawMovies) {
    // Combine fields into a single rich text representation for embedding
    const richText = `Title: ${movie.title}. Genres: ${movie.genres.join(', ')}. Plot: ${movie.overview}`;

    console.log(`  └─ Embedding "${movie.title}"...`);
    const vector = await getEmbedding(richText);

    moviesWithVectors.push({
      ...movie,
      vector,
    });
  }

  console.log('💾 Writing embedded dataset to movies_with_vectors.json...');
  await fs.writeFile(outputPath, JSON.stringify(moviesWithVectors, null, 2), 'utf-8');

  console.log('✨ Ingestion Complete! Dataset is ready for search.');
}

runIngestion().catch(console.error);