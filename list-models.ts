import { GoogleGenAI } from '@google/genai';
import { config } from 'dotenv';
import { resolve } from 'path';

config({ path: resolve(process.cwd(), '.env') });
config({ path: resolve(process.cwd(), '..', '.env') });

async function listModels() {
    const apiKey = process.env.GOOGLE_GEMINI_API_KEY || process.env.GOOGLE_GENERATIVE_AI_API_KEY;
    if (!apiKey) {
        console.error('API key not found');
        return;
    }

    try {
        const genAI = new GoogleGenAI({ apiKey });
        const modelList = await genAI.models.list();
        console.log('Available models:');
        for await (const model of modelList) {
            console.log(`- ${model.name} (Supported: ${model.supportedGenerationMethods})`);
        }
    } catch (error) {
        console.error('Error listing models:', error);
    }
}

listModels();
