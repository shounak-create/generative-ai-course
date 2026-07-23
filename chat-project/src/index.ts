import { GoogleGenAI } from '@google/genai';
import * as readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';

const apiKey = process.env.GEMINI_API_KEY;

if (!apiKey) {
  throw new Error('GEMINI_API_KEY environment variable is missing.');
}

const ai = new GoogleGenAI({ apiKey });

async function startTerminalChat() {
  const rl = readline.createInterface({ input, output });

  const chat = ai.chats.create({
    model: 'gemini-3.5-flash',
    config: {
      systemInstruction: 'You are a helpful, witty AI assistant in a terminal chat app.',
    },
  });

  console.log('🤖 Terminal Chat Initialized! Type "exit" or "quit" to stop.\n');

  while (true) {
    const userInput = await rl.question('You: ');

    // Check for exit commands
    if (userInput.trim().toLowerCase() === 'exit' || userInput.trim().toLowerCase() === 'quit') {
      console.log('Goodbye!');
      rl.close();
      break;
    }

    if (!userInput.trim()) continue;

    try {
      process.stdout.write('Gemini: ');

      const responseStream = await chat.sendMessageStream({
        message: userInput,
      });

      for await (const chunk of responseStream) {
        if (chunk.text) {
          // process.stdout.write('Gemini: ');
          process.stdout.write(chunk.text);
        }
      }

      console.log('\n'); // Add extra line break between turns
    } catch (error) {
      console.error('\nAn error occurred:', error);
    }
  }
}

startTerminalChat();