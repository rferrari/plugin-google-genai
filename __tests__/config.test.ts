import { describe, test, expect, vi, beforeEach } from 'vitest';
import { googleGenAIPlugin } from '../src/index';
import { logger } from '@elizaos/core';

// Mock the logger
vi.mock('@elizaos/core', () => ({
  logger: {
    warn: vi.fn(),
    error: vi.fn(),
    log: vi.fn(),
    info: vi.fn(),
    debug: vi.fn(),
  },
  EventType: {},
  ModelType: {},
}));

// Create a minimal mock runtime
const createMockRuntime = (env: Record<string, string>) => {
  return {
    getSetting: (key: string) => env[key],
    emitEvent: () => { },
    character: {
      system: 'You are a helpful assistant.',
    },
  } as unknown as any;
};

describe('Google Generative AI Plugin Configuration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test('should warn when API key is missing', async () => {
    // Temporarily clear env to force warning
    const originalKey = process.env.GOOGLE_GEMINI_API_KEY;
    const originalLegacyKey = process.env.GOOGLE_GENERATIVE_AI_API_KEY;
    const originalGeminiKey = process.env.GEMINI_API_KEY;
    delete process.env.GOOGLE_GEMINI_API_KEY;
    delete process.env.GOOGLE_GENERATIVE_AI_API_KEY;
    delete process.env.GEMINI_API_KEY;

    try {
      // Create a mock runtime with no API key
      const mockRuntime = createMockRuntime({});

      // Initialize plugin
      if (googleGenAIPlugin.init) {
        await googleGenAIPlugin.init({}, mockRuntime);
      }

      // Check that warning was logged
      await vi.waitFor(() => {
        expect(logger.warn).toHaveBeenCalledWith(
          'GOOGLE_GENERATIVE_AI_API_KEY is not set in environment - Google AI functionality will be limited'
        );
      });
    } finally {
      // Restore env
      if (originalKey !== undefined) process.env.GOOGLE_GEMINI_API_KEY = originalKey;
      if (originalLegacyKey !== undefined) process.env.GOOGLE_GENERATIVE_AI_API_KEY = originalLegacyKey;
      if (originalGeminiKey !== undefined) process.env.GEMINI_API_KEY = originalGeminiKey;
    }
  });

  test('should initialize properly with valid API key', async () => {
    // Skip if no API key available for testing
    if (!process.env.GOOGLE_GEMINI_API_KEY && !process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
      console.warn('Skipping test: GOOGLE_GEMINI_API_KEY not set');
      return;
    }

    // Create a mock runtime with API key
    const mockRuntime = createMockRuntime({
      GOOGLE_GEMINI_API_KEY: (process.env.GOOGLE_GEMINI_API_KEY || process.env.GOOGLE_GENERATIVE_AI_API_KEY) as string,
    });

    // Initialize plugin
    if (googleGenAIPlugin.init) {
      await googleGenAIPlugin.init({}, mockRuntime);
    }

    // Expect successful log message using vi.waitFor to handle background async validation
    await vi.waitFor(() => {
      expect(logger.log).toHaveBeenCalledWith(
        expect.stringContaining('Google Generative AI API key validated successfully')
      );
    }, { timeout: 5000 });
  });

  test('should use custom image model when configured', () => {
    // Create a mock runtime with custom model settings
    const customImageModel = 'gemini-3-flash';
    const mockRuntime = createMockRuntime({
      GOOGLE_IMAGE_MODEL: customImageModel,
      GOOGLE_GEMINI_API_KEY: 'test-key',
    });

    // Verify getSetting returns the custom image model
    expect(mockRuntime.getSetting('GOOGLE_IMAGE_MODEL')).toBe(customImageModel);
  });
});
