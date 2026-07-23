import { GoogleGenAI } from "@google/genai";
import { encoding_for_model } from "tiktoken";

const encoder = encoding_for_model("gpt-4");

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

async function main() {
  const response = await ai.models.generateContent({
    model: "gemini-3.5-flash-lite",
    contents: "How tall is Mount Everest?",
  });

  console.log(response.text);
}

main();