import { GoogleGenAI, HarmCategory, HarmBlockThreshold } from '@google/genai';
import type { IAgentRuntime } from '@elizaos/core';
import { logger } from '@elizaos/core';

/**
 * Retrieves a configuration setting from the runtime, falling back to environment variables or a default value if not found.
 *
 * @param key - The name of the setting to retrieve.
 * @param defaultValue - The value to return if the setting is not found in the runtime or environment.
 * @returns The resolved setting value, or {@link defaultValue} if not found.
 */
export function getSetting(
  runtime: IAgentRuntime,
  key: string,
  defaultValue?: string
): string | undefined {
  const value = runtime.getSetting(key);
  return value !== undefined && value !== null ? String(value) : (process.env[key] ?? defaultValue);
}

/**
 * Helper function to get the API key for Google Gemini
 *
 * @param runtime The runtime context
 * @returns The configured API key
 */
export function getApiKey(runtime: IAgentRuntime): string | undefined {
  return (
    getSetting(runtime, 'GOOGLE_GEMINI_API_KEY') ||
    getSetting(runtime, 'GEMINI_API_KEY') ||
    getSetting(runtime, 'GOOGLE_GENERATIVE_AI_API_KEY') ||
    undefined
  );
}

/**
 * Helper function to get the small model name with fallbacks
 *
 * @param runtime The runtime context
 * @returns The configured small model name
 */
export function getSmallModel(runtime: IAgentRuntime): string {
  return (
    getSetting(runtime, 'GOOGLE_SMALL_MODEL') ??
    'gemini-3-flash-preview'
  );
}

/**
 * Helper function to get the large model name with fallbacks
 *
 * @param runtime The runtime context
 * @returns The configured large model name
 */
export function getLargeModel(runtime: IAgentRuntime): string {
  return (
    getSetting(runtime, 'GOOGLE_LARGE_MODEL') ??
    'gemini-3.1-pro-preview'
  );
}

/**
 * Helper function to get the image model name with fallbacks
 *
 * @param runtime The runtime context
 * @returns The configured image model name
 */
export function getImageModel(runtime: IAgentRuntime): string {
  return (
    getSetting(runtime, 'GOOGLE_IMAGE_MODEL') ??
    'gemini-3-flash-preview'
  );
}

/**
 * Helper function to get the embedding model name with fallbacks
 *
 * @param runtime The runtime context
 * @returns The configured embedding model name
 */
export function getEmbeddingModel(runtime: IAgentRuntime): string {
  return (
    getSetting(runtime, 'GOOGLE_EMBEDDING_MODEL') ??
    'gemini-embedding-001'
  );
}

/**
 * Create a Google Gemini client instance with proper configuration
 *
 * @param runtime The runtime context
 * @returns Configured Google Gemini instance
 */
export function createGoogleGenAI(runtime: IAgentRuntime): GoogleGenAI | null {
  const apiKey = getApiKey(runtime);
  if (!apiKey) {
    logger.error('Google Generative AI API Key is missing');
    return null;
  }

  return new GoogleGenAI({ apiKey });
}

/**
 * Modernized alias for client factory
 */
export const createGoogleGemini = createGoogleGenAI;

/**
 * Convert safety settings to Google format
 */
export function getSafetySettings() {
  return [
    {
      category: HarmCategory.HARM_CATEGORY_HARASSMENT,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
      category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
      category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
      category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
  ];
}
