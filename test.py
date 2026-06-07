"""
AI Message Module: Uses v98store API with OpenAI SDK
Endpoint: https://v98store.com/v1
Model: gpt-5-mini (for content generation only)
"""

from openai import OpenAI

API_KEY = "sk-nqI9IjzU3qaMYuf5k6I5xSiHY5ujjY2R4flVhgMYi6I7spM3"
MODEL = "gpt-5-mini"
SYSTEM_PROMPT = "Do not use any emojis in your responses. Be concise."

client = OpenAI(
    base_url="https://v98store.com/v1",
    api_key=API_KEY,
)


def send_to_ai(question, max_tokens=1000, temperature=0.7, messages=None):
    if messages:
        has_system = any(m["role"] == "system" for m in messages)
        if has_system:
            msgs = [
                {**m, "content": f"{m['content']}\n{SYSTEM_PROMPT}"} if m["role"] == "system" else m
                for m in messages
            ]
        else:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    else:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

    completion = client.chat.completions.create(
        model=MODEL,
        messages=msgs,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    text = completion.choices[0].message.content
    if not text:
        raise ValueError("API response missing content")
    return text


if __name__ == "__main__":
    questions = [
        "What is the capital of France?",
        "Explain quicksort in 2 sentences.",
        "What is 123 * 456?",
    ]

    for q in questions:
        print(f"Q: {q}")
        answer = send_to_ai(q)
        print(f"A: {answer}\n")
