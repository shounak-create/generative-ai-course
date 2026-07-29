import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import * as dotenv from "dotenv";

dotenv.config();

async function runFirstChain() {
  // 1. Initialize the LLM via LangChain's Google integration
  const model = new ChatGoogleGenerativeAI({
    model: "gemini-3.5-flash",
    apiKey: process.env.GEMINI_API_KEY,
    temperature: 0.7,
    verbose:true,
  });

  // 2. Define a Prompt Template with dynamic variables
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a film critic AI specializing in quick movie elevator pitches.",
    ],
    [
      "user",
      "Give me a 2-sentence movie pitch for a film about {genre} featuring {theme}.",
    ],
  ]);

  // 3. Define an Output Parser to extract raw text string directly
  const outputParser = new StringOutputParser();

  // 4. Combine them into a Chain using LCEL (.pipe)
  const chain = prompt.pipe(model).pipe(outputParser);

  // 5. Invoke the chain passing dynamic input values
  console.log("Running LangChain pipeline...\n");

  const response = await chain.invoke({
    genre: "Sci-Fi Thriller",
    theme: "time travel and corporate espionage",
  });

  console.log("--- AI Response ---");
  console.log(response);
}

runFirstChain().catch(console.error);