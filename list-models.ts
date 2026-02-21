import { GoogleGenAI } from '@google/genai';
import { config } from 'dotenv';
import { resolve } from 'path';

config({ path: resolve(process.cwd(), '.env') });
config({ path: resolve(process.cwd(), '..', '.env') });

async function listModels() {
    const apiKey = process.env.GOOGLE_GEMINI_API_KEY || process.env.GOOGLE_GENERATIVE_AI_API_KEY;
    if (!apiKey) {
        console.error(
            'API key not found. Set GOOGLE_GEMINI_API_KEY (or legacy GOOGLE_GENERATIVE_AI_API_KEY) in your environment.'
        );
        return;
    }

    try {
        const genAI = new GoogleGenAI({ apiKey });
        const modelList = await genAI.models.list();
        console.log('Available models:');
        for await (const model of modelList) {
            console.log(`- ${model.name} (${model.description || 'No description'})`);
        }
    } catch (error) {
        console.error('Error listing models:', error);
    }
}

listModels();
