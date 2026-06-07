/**
 * AI Message Module: Uses v98store API with OpenAI SDK
 * Endpoint: https://v98store.com/v1
 * Model: gpt-5-mini (for content generation only)
 * Note: Verification challenges use gemini-3-flash-preview in batch_post.py
 */

import OpenAI from 'openai';

/** @type {string} v98store API key */
const API_KEY = "sk-nqI9IjzU3qaMYuf5k6I5xSiHY5ujjY2R4flVhgMYi6I7spM3";

/** @type {string} Model name to use */
const MODEL = "gpt-5-mini";

/** @type {string} System prompt to control AI behavior */
const SYSTEM_PROMPT = "Do not use any emojis in your responses. Be concise.";

/** Initialize OpenAI client with v98store base URL */
const client = new OpenAI({
    baseURL: "https://v98store.com/v1",
    apiKey: API_KEY,
});

/**
 * Send a question to v98store AI and get a response
 * @param {string} question - User question or message content
 * @param {Object} [options] - Optional parameters
 * @param {number} [options.maxTokens=1000] - Max tokens to generate
 * @param {number} [options.temperature=0.7] - Sampling temperature 0-2
 * @param {Array<{role: string, content: string}>} [options.messages] - Custom conversation history
 * @returns {Promise<string>} AI response text content
 * @throws {Error} When request fails or API returns error
 */
async function sendToAI(question, options = {}) {
    const {
        maxTokens = 1000,
        temperature = 0.7,
        messages: customMessages,
    } = options;

    try {
        // Build messages array with system prompt
        let messages;
        const systemMessage = { role: "system", content: SYSTEM_PROMPT };

        if (Array.isArray(customMessages) && customMessages.length > 0) {
            // Check if custom messages already have a system message
            const hasSystemMessage = customMessages.some(msg => msg.role === "system");
            if (hasSystemMessage) {
                // Append instruction to existing system message
                messages = customMessages.map(msg =>
                    msg.role === "system"
                        ? { ...msg, content: `${msg.content}\n${SYSTEM_PROMPT}` }
                        : msg
                );
            } else {
                // Prepend system message
                messages = [systemMessage, ...customMessages];
            }
        } else {
            // Simple question - wrap in user message with system prompt
            messages = [systemMessage, { role: "user", content: question }];
        }

        // Call v98store API via OpenAI SDK
        const completion = await client.chat.completions.create({
            model: MODEL,
            messages: messages,
            max_tokens: maxTokens,
            temperature: temperature,
        });

        // Get generated text
        const text = completion.choices[0]?.message?.content;

        if (!text) {
            throw new Error("API response missing content");
        }

        return text;
    } catch (error) {
        // Provide more detailed error info
        if (error.message) {
            throw new Error(`v98store API request failed: ${error.message}`);
        }
        throw new Error(`v98store API request failed: ${JSON.stringify(error)}`);
    }
}

/**
 * Get current config (redacted API key for display)
 * @returns {{ model: string, apiKey: string }}
 */
function getConfig() {
    return {
        model: MODEL,
        apiKey: API_KEY.substring(0, 10) + "..." // Redacted display
    };
}

export { sendToAI, getConfig, API_KEY, MODEL };
