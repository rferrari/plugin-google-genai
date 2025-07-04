import { GoogleGenAI, HarmCategory, HarmBlockThreshold } from '@google/genai';
import type {
  IAgentRuntime,
  ModelTypeName,
  ObjectGenerationParams,
  Plugin,
  GenerateTextParams,
  ImageDescriptionParams,
  TextEmbeddingParams,
} from '@elizaos/core';
import { EventType, logger, ModelType } from '@elizaos/core';
import { fetch } from 'undici';

/**
 * Retrieves a configuration setting from the runtime, falling back to environment variables or a default value if not found.
 *
 * @param key - The name of the setting to retrieve.
 * @param defaultValue - The value to return if the setting is not found in the runtime or environment.
 * @returns The resolved setting value, or {@link defaultValue} if not found.
 */
function getSetting(
  runtime: IAgentRuntime,
  key: string,
  defaultValue?: string
): string | undefined {
  return runtime.getSetting(key) ?? process.env[key] ?? defaultValue;
}

/**
 * Helper function to get the API key for Google AI
 *
 * @param runtime The runtime context
 * @returns The configured API key
 */
function getApiKey(runtime: IAgentRuntime): string | undefined {
  return getSetting(runtime, 'GOOGLE_GENERATIVE_AI_API_KEY');
}

/**
 * Helper function to get the small model name with fallbacks
 *
 * @param runtime The runtime context
 * @returns The configured small model name
 */
function getSmallModel(runtime: IAgentRuntime): string {
  return (
    getSetting(runtime, 'GOOGLE_SMALL_MODEL') ??
    getSetting(runtime, 'SMALL_MODEL', 'gemini-2.0-flash-001') ??
    'gemini-2.0-flash-001'
  );
}

/**
 * Helper function to get the large model name with fallbacks
 *
 * @param runtime The runtime context
 * @returns The configured large model name
 */
function getLargeModel(runtime: IAgentRuntime): string {
  return (
    getSetting(runtime, 'GOOGLE_LARGE_MODEL') ??
    getSetting(runtime, 'LARGE_MODEL', 'gemini-2.5-pro-preview-03-25') ??
    'gemini-2.5-pro-preview-03-25'
  );
}

/**
 * Helper function to get the image model name with fallbacks
 *
 * @param runtime The runtime context
 * @returns The configured image model name
 */
function getImageModel(runtime: IAgentRuntime): string {
  return (
    getSetting(runtime, 'GOOGLE_IMAGE_MODEL') ??
    getSetting(runtime, 'IMAGE_MODEL', 'gemini-2.5-pro-preview-03-25') ??
    'gemini-2.5-pro-preview-03-25'
  );
}

/**
 * Helper function to get the embedding model name with fallbacks
 *
 * @param runtime The runtime context
 * @returns The configured embedding model name
 */
function getEmbeddingModel(runtime: IAgentRuntime): string {
  return (
    getSetting(runtime, 'GOOGLE_EMBEDDING_MODEL', 'text-embedding-004') ?? 'text-embedding-004'
  );
}

/**
 * Create a Google Generative AI client instance with proper configuration
 *
 * @param runtime The runtime context
 * @returns Configured Google Generative AI instance
 */
function createGoogleGenAI(runtime: IAgentRuntime): GoogleGenAI | null {
  const apiKey = getApiKey(runtime);
  if (!apiKey) {
    logger.error('Google Generative AI API Key is missing');
    return null;
  }

  return new GoogleGenAI({ apiKey });
}

/**
 * Convert safety settings to Google format
 */
function getSafetySettings() {
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

/**
 * Emits a model usage event
 * @param runtime The runtime context
 * @param type The model type
 * @param prompt The prompt used
 * @param usage The usage data
 */
function emitModelUsageEvent(
  runtime: IAgentRuntime,
  type: ModelTypeName,
  prompt: string,
  usage: { promptTokens: number; completionTokens: number; totalTokens: number }
) {
  runtime.emitEvent(EventType.MODEL_USED, {
    provider: 'google',
    type,
    prompt,
    tokens: {
      prompt: usage.promptTokens,
      completion: usage.completionTokens,
      total: usage.totalTokens,
    },
  });
}

/**
 * Helper function to count tokens for a given text (estimation)
 */
async function countTokens(text: string): Promise<number> {
  // Rough estimation: ~1 token per 4 characters
  return Math.ceil(text.length / 4);
}

/**
 * Helper function to generate objects using specified model type
 */
async function generateObjectByModelType(
  runtime: IAgentRuntime,
  params: ObjectGenerationParams,
  modelType: string,
  getModelFn: (runtime: IAgentRuntime) => string
): Promise<any> {
  const genAI = createGoogleGenAI(runtime);
  if (!genAI) {
    throw new Error('Google Generative AI client not initialized');
  }

  const modelName = getModelFn(runtime);
  const temperature = params.temperature ?? 0.1;

  logger.info(`Using ${modelType} model: ${modelName}`);

  try {
    // Add schema instructions to prompt if provided
    let enhancedPrompt = params.prompt;
    if (params.schema) {
      enhancedPrompt += `\n\nPlease respond with a JSON object that follows this schema:\n${JSON.stringify(params.schema, null, 2)}`;
    }

    const response = await genAI.models.generateContent({
      model: modelName,
      contents: enhancedPrompt,
      config: {
        temperature,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: 8192,
        responseMimeType: 'application/json',
        safetySettings: getSafetySettings(),
      },
    });

    const text = response.text || '';

    // Count tokens for usage tracking
    const promptTokens = await countTokens(enhancedPrompt);
    const completionTokens = await countTokens(text);

    emitModelUsageEvent(runtime, modelType as ModelTypeName, params.prompt, {
      promptTokens,
      completionTokens,
      totalTokens: promptTokens + completionTokens,
    });

    try {
      return JSON.parse(text);
    } catch (parseError) {
      logger.error('Failed to parse JSON response:', parseError);
      // Try to extract JSON from the response
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        try {
          return JSON.parse(jsonMatch[0]);
        } catch (secondParseError) {
          throw new Error('Failed to parse JSON from response');
        }
      }
      throw parseError;
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    logger.error(`[generateObject] Error: ${message}`);
    throw error;
  }
}

/**
 * Defines the Google Generative AI plugin with its name, description, and configuration options.
 * @type {Plugin}
 *
 * Available models as of March 2025:
 * - gemini-2.0-flash-001: Fast, efficient model for everyday tasks
 * - gemini-2.5-pro-exp-03-25: Latest experimental model with advanced reasoning (March 25, 2025)
 * - gemini-2.5-pro-preview-05-06: Preview version from Google I/O 2025
 * - gemini-2.5-pro: General model name for Gemini 2.5 Pro
 * - text-embedding-004: For text embeddings
 */
export const googleGenAIPlugin: Plugin = {
  name: 'google-genai',
  description: 'Google Generative AI plugin for Gemini models',
  config: {
    GOOGLE_GENERATIVE_AI_API_KEY: process.env.GOOGLE_GENERATIVE_AI_API_KEY,
    GOOGLE_SMALL_MODEL: process.env.GOOGLE_SMALL_MODEL,
    GOOGLE_LARGE_MODEL: process.env.GOOGLE_LARGE_MODEL,
    GOOGLE_IMAGE_MODEL: process.env.GOOGLE_IMAGE_MODEL,
    GOOGLE_EMBEDDING_MODEL: process.env.GOOGLE_EMBEDDING_MODEL,
    SMALL_MODEL: process.env.SMALL_MODEL,
    LARGE_MODEL: process.env.LARGE_MODEL,
    IMAGE_MODEL: process.env.IMAGE_MODEL,
  },
  async init(_config, runtime) {
    try {
      const apiKey = getApiKey(runtime);
      if (!apiKey) {
        logger.warn(
          'GOOGLE_GENERATIVE_AI_API_KEY is not set in environment - Google AI functionality will be limited'
        );
        return;
      }

      // Test the API key by listing models
      try {
        const genAI = new GoogleGenAI({ apiKey });
        const modelList = await genAI.models.list();
        const models = [];
        for await (const model of modelList) {
          models.push(model);
        }
        logger.log(`Google AI API key validated successfully. Available models: ${models.length}`);
      } catch (fetchError: unknown) {
        const message = fetchError instanceof Error ? fetchError.message : String(fetchError);
        logger.warn(`Error validating Google AI API key: ${message}`);
      }
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      logger.warn(
        `Google AI plugin configuration issue: ${message} - You need to configure the GOOGLE_GENERATIVE_AI_API_KEY in your environment variables`
      );
    }
  },
  models: {
    [ModelType.TEXT_SMALL]: async (
      runtime: IAgentRuntime,
      { prompt, stopSequences = [] }: GenerateTextParams
    ) => {
      const genAI = createGoogleGenAI(runtime);
      if (!genAI) {
        throw new Error('Google Generative AI client not initialized');
      }

      const modelName = getSmallModel(runtime);
      const temperature = 0.7;
      const maxOutputTokens = 8192;

      logger.log(`[TEXT_SMALL] Using model: ${modelName}`);
      logger.debug(`[TEXT_SMALL] Prompt: ${prompt}`);

      try {
        const systemInstruction = runtime.character.system || undefined;
        const response = await genAI.models.generateContent({
          model: modelName,
          contents: prompt,
          config: {
            temperature,
            topK: 40,
            topP: 0.95,
            maxOutputTokens,
            stopSequences,
            safetySettings: getSafetySettings(),
            ...(systemInstruction && { systemInstruction }),
          },
        });

        const text = response.text || '';

        // Count tokens for usage tracking
        const promptTokens = await countTokens(prompt);
        const completionTokens = await countTokens(text);

        emitModelUsageEvent(runtime, ModelType.TEXT_SMALL, prompt, {
          promptTokens,
          completionTokens,
          totalTokens: promptTokens + completionTokens,
        });

        return text;
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        logger.error(`[TEXT_SMALL] Error: ${message}`);
        throw error;
      }
    },
    [ModelType.TEXT_LARGE]: async (
      runtime: IAgentRuntime,
      {
        prompt,
        stopSequences = [],
        maxTokens = 8192,
        temperature = 0.7,
        frequencyPenalty = 0.7,
        presencePenalty = 0.7,
      }: GenerateTextParams
    ) => {
      const genAI = createGoogleGenAI(runtime);
      if (!genAI) {
        throw new Error('Google Generative AI client not initialized');
      }

      const modelName = getLargeModel(runtime);

      logger.log(`[TEXT_LARGE] Using model: ${modelName}`);
      logger.debug(`[TEXT_LARGE] Prompt: ${prompt}`);

      try {
        const systemInstruction = runtime.character.system || undefined;
        const response = await genAI.models.generateContent({
          model: modelName,
          contents: prompt,
          config: {
            temperature,
            topK: 40,
            topP: 0.95,
            maxOutputTokens: maxTokens,
            stopSequences,
            safetySettings: getSafetySettings(),
            ...(systemInstruction && { systemInstruction }),
          },
        });

        const text = response.text || '';

        // Count tokens for usage tracking
        const promptTokens = await countTokens(prompt);
        const completionTokens = await countTokens(text);

        emitModelUsageEvent(runtime, ModelType.TEXT_LARGE, prompt, {
          promptTokens,
          completionTokens,
          totalTokens: promptTokens + completionTokens,
        });

        return text;
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        logger.error(`[TEXT_LARGE] Error: ${message}`);
        throw error;
      }
    },
    [ModelType.TEXT_EMBEDDING]: async (
      runtime: IAgentRuntime,
      params: TextEmbeddingParams | string | null
    ): Promise<number[]> => {
      const genAI = createGoogleGenAI(runtime);
      if (!genAI) {
        throw new Error('Google Generative AI client not initialized');
      }

      const embeddingModelName = getEmbeddingModel(runtime);
      logger.debug(`[TEXT_EMBEDDING] Using model: ${embeddingModelName}`);

      // Handle null case for initialization
      if (params === null) {
        logger.debug('Creating test embedding for initialization');
        // Return 768-dimensional vector for text-embedding-004
        const dimension = 768;
        const testVector = Array(dimension).fill(0);
        testVector[0] = 0.1;
        return testVector;
      }

      // Extract text from params
      let text: string;
      if (typeof params === 'string') {
        text = params;
      } else if (typeof params === 'object' && params.text) {
        text = params.text;
      } else {
        logger.warn('Invalid input format for embedding');
        const dimension = 768;
        const fallbackVector = Array(dimension).fill(0);
        fallbackVector[0] = 0.2;
        return fallbackVector;
      }

      if (!text.trim()) {
        logger.warn('Empty text for embedding');
        const dimension = 768;
        const emptyVector = Array(dimension).fill(0);
        emptyVector[0] = 0.3;
        return emptyVector;
      }

      try {
        const response = await genAI.models.embedContent({
          model: embeddingModelName,
          contents: text,
        });

        const embedding = response.embeddings?.[0]?.values || [];

        // Count tokens for usage tracking
        const promptTokens = await countTokens(text);

        emitModelUsageEvent(runtime, ModelType.TEXT_EMBEDDING, text, {
          promptTokens,
          completionTokens: 0,
          totalTokens: promptTokens,
        });

        logger.log(`Got embedding with length ${embedding.length}`);
        return embedding;
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        logger.error(`Error generating embedding: ${message}`);
        // Return error vector
        const dimension = 768;
        const errorVector = Array(dimension).fill(0);
        errorVector[0] = 0.6;
        return errorVector;
      }
    },
    [ModelType.IMAGE_DESCRIPTION]: async (
      runtime: IAgentRuntime,
      params: ImageDescriptionParams | string
    ) => {
      const genAI = createGoogleGenAI(runtime);
      if (!genAI) {
        throw new Error('Google Generative AI client not initialized');
      }

      let imageUrl: string;
      let promptText: string | undefined;
      const modelName = getImageModel(runtime);
      logger.log(`[IMAGE_DESCRIPTION] Using model: ${modelName}`);

      if (typeof params === 'string') {
        imageUrl = params;
        promptText = 'Please analyze this image and provide a title and detailed description.';
      } else {
        imageUrl = params.imageUrl;
        promptText =
          params.prompt ||
          'Please analyze this image and provide a title and detailed description.';
      }

      try {
        // Fetch image data
        const imageResponse = await fetch(imageUrl);
        if (!imageResponse.ok) {
          throw new Error(`Failed to fetch image: ${imageResponse.statusText}`);
        }

        const imageData = await imageResponse.arrayBuffer();
        const base64Image = Buffer.from(imageData).toString('base64');

        // Determine MIME type from URL or response headers
        const contentType = imageResponse.headers.get('content-type') || 'image/jpeg';

        const response = await genAI.models.generateContent({
          model: modelName,
          contents: [
            {
              role: 'user',
              parts: [
                { text: promptText },
                {
                  inlineData: {
                    mimeType: contentType,
                    data: base64Image,
                  },
                },
              ],
            },
          ],
          config: {
            temperature: 0.7,
            topK: 40,
            topP: 0.95,
            maxOutputTokens: 8192,
            safetySettings: getSafetySettings(),
          },
        });

        const responseText = response.text || '';

        logger.log('Received response for image description');

        // Check if a custom prompt was provided
        const isCustomPrompt =
          typeof params === 'object' &&
          params.prompt &&
          params.prompt !==
            'Please analyze this image and provide a title and detailed description.';

        // If custom prompt is used, return the raw content
        if (isCustomPrompt) {
          return responseText;
        }

        // Try to parse the response as JSON first
        try {
          const jsonResponse = JSON.parse(responseText);
          if (jsonResponse.title && jsonResponse.description) {
            return jsonResponse;
          }
        } catch (e) {
          // If not valid JSON, process as text
          logger.debug(`Parsing as JSON failed, processing as text: ${e}`);
        }

        // Extract title and description from text format
        const titleMatch = responseText.match(/title[:\s]+(.+?)(?:\n|$)/i);
        const title = titleMatch?.[1]?.trim() || 'Image Analysis';
        const description = responseText.replace(/title[:\s]+(.+?)(?:\n|$)/i, '').trim();

        return { title, description };
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        logger.error(`Error analyzing image: ${message}`);
        return {
          title: 'Failed to analyze image',
          description: `Error: ${message}`,
        };
      }
    },
    [ModelType.OBJECT_SMALL]: async (runtime: IAgentRuntime, params: ObjectGenerationParams) => {
      return generateObjectByModelType(runtime, params, ModelType.OBJECT_SMALL, getSmallModel);
    },
    [ModelType.OBJECT_LARGE]: async (runtime: IAgentRuntime, params: ObjectGenerationParams) => {
      return generateObjectByModelType(runtime, params, ModelType.OBJECT_LARGE, getLargeModel);
    },
  },
  tests: [
    {
      name: 'google_genai_plugin_tests',
      tests: [
        {
          name: 'google_test_api_key_validation',
          fn: async (runtime: IAgentRuntime) => {
            const apiKey = getApiKey(runtime);
            if (!apiKey) {
              throw new Error('GOOGLE_GENERATIVE_AI_API_KEY not set');
            }
            const genAI = new GoogleGenAI({ apiKey });
            const modelList = await genAI.models.list();
            const models = [];
            for await (const model of modelList) {
              models.push(model);
            }
            logger.log('Available models:', models.length);
          },
        },
        {
          name: 'google_test_text_embedding',
          fn: async (runtime: IAgentRuntime) => {
            try {
              const embedding = await runtime.useModel(ModelType.TEXT_EMBEDDING, {
                text: 'Hello, world!',
              });
              logger.log('Embedding dimension:', embedding.length);
              if (embedding.length === 0) {
                throw new Error('Failed to generate embedding');
              }
            } catch (error: unknown) {
              const message = error instanceof Error ? error.message : String(error);
              logger.error(`Error in test_text_embedding: ${message}`);
              throw error;
            }
          },
        },
        {
          name: 'google_test_text_small',
          fn: async (runtime: IAgentRuntime) => {
            try {
              const text = await runtime.useModel(ModelType.TEXT_SMALL, {
                prompt: 'What is the nature of reality in 10 words?',
              });
              if (text.length === 0) {
                throw new Error('Failed to generate text');
              }
              logger.log('Generated with TEXT_SMALL:', text);
            } catch (error: unknown) {
              const message = error instanceof Error ? error.message : String(error);
              logger.error(`Error in test_text_small: ${message}`);
              throw error;
            }
          },
        },
        {
          name: 'google_test_text_large',
          fn: async (runtime: IAgentRuntime) => {
            try {
              const text = await runtime.useModel(ModelType.TEXT_LARGE, {
                prompt: 'Explain quantum mechanics in simple terms.',
              });
              if (text.length === 0) {
                throw new Error('Failed to generate text');
              }
              logger.log('Generated with TEXT_LARGE:', text.substring(0, 100) + '...');
            } catch (error: unknown) {
              const message = error instanceof Error ? error.message : String(error);
              logger.error(`Error in test_text_large: ${message}`);
              throw error;
            }
          },
        },
        {
          name: 'google_test_image_description',
          fn: async (runtime: IAgentRuntime) => {
            try {
              const result = await runtime.useModel(
                ModelType.IMAGE_DESCRIPTION,
                'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Vitalik_Buterin_TechCrunch_London_2015_%28cropped%29.jpg/537px-Vitalik_Buterin_TechCrunch_London_2015_%28cropped%29.jpg'
              );

              if (
                result &&
                typeof result === 'object' &&
                'title' in result &&
                'description' in result
              ) {
                logger.log('Image description:', result);
              } else {
                logger.error('Invalid image description result format:', result);
              }
            } catch (error: unknown) {
              const message = error instanceof Error ? error.message : String(error);
              logger.error(`Error in test_image_description: ${message}`);
              throw error;
            }
          },
        },
        {
          name: 'google_test_object_generation',
          fn: async (runtime: IAgentRuntime) => {
            try {
              const schema = {
                type: 'object',
                properties: {
                  name: { type: 'string' },
                  age: { type: 'number' },
                  hobbies: { type: 'array', items: { type: 'string' } },
                },
                required: ['name', 'age', 'hobbies'],
              };

              const result = await runtime.useModel(ModelType.OBJECT_SMALL, {
                prompt: 'Generate a person profile with name, age, and hobbies.',
                schema,
              });

              logger.log('Generated object:', result);

              if (!result.name || !result.age || !result.hobbies) {
                throw new Error('Generated object missing required fields');
              }
            } catch (error: unknown) {
              const message = error instanceof Error ? error.message : String(error);
              logger.error(`Error in test_object_generation: ${message}`);
              throw error;
            }
          },
        },
      ],
    },
  ],
};

export default googleGenAIPlugin;
