import { GoogleGenAI } from '@google/genai';
import * as fs from 'node:fs';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

async function generateImage() {
  const response = await ai.models.generateImages({
    model: 'imagen-4.0-generate-001',
    prompt: 'A futuristic cybernetic cat sitting on a neon-lit rooftop, digital art',
    config: {
      numberOfImages: 1,
      outputMimeType: 'image/jpeg',
      aspectRatio: '1:1', // Options: '1:1', '3:4', '4:3', '16:9', etc.
    },
  });

  const base64ImageBytes = response.generatedImages[0].image.imageBytes;
  const buffer = Buffer.from(base64ImageBytes, 'base64');
  
  // Save to local file
  fs.writeFileSync('output.jpg', buffer);
  console.log('🖼️ Image saved to output.jpg!');
}

generateImage();