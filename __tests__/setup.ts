import { config } from 'dotenv';
import { beforeAll } from 'vitest';
import { resolve } from 'path';

// Load environment variables from .env file
beforeAll(() => {
  // Try current directory first, then parent directory (root)
  config({ path: resolve(process.cwd(), '.env') });
  config({ path: resolve(process.cwd(), '..', '.env') });

  // Check if required environment variables are set
  if (!process.env.GOOGLE_GEMINI_API_KEY && !process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
    console.warn('⚠️  GOOGLE_GEMINI_API_KEY not found in .env file. Tests may fail.');
  }
});
