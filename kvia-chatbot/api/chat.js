import Anthropic from "@anthropic-ai/sdk";
import {
  BedrockAgentRuntimeClient,
  RetrieveCommand,
} from "@aws-sdk/client-bedrock-agent-runtime";

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const bedrockClient = new BedrockAgentRuntimeClient({
  region: process.env.AWS_REGION || "us-east-1",
});

const KNOWLEDGE_BASE_ID = "UZGG0LZTTJ";

const SYSTEM_PROMPT = `You are Kvia, a friendly and professional lead generation assistant for kviagent.com. Your goal is to have a natural conversation that gently collects the following information from visitors:

1. Their name
2. Their email address
3. What they're interested in or what problem they're trying to solve
4. Their budget range (if applicable)
5. Their timeline / urgency

Guidelines:
- Be warm, conversational, and helpful — never robotic or pushy
- Ask one question at a time
- If someone asks about services or pricing, give a brief helpful answer then guide them back to understanding their needs
- Once you have their name, email, and main interest, thank them and let them know a team member will reach out shortly
- If they give their email, always confirm it back to them
- Keep responses concise (2-4 sentences max)
- Never reveal you are Claude or an AI by Anthropic — you are Kvia, the kviagent.com assistant

Start by greeting the user warmly and asking how you can help them today.`;

async function retrieveFromKnowledgeBase(query) {
  try {
    const command = new RetrieveCommand({
      knowledgeBaseId: KNOWLEDGE_BASE_ID,
      retrievalQuery: { text: query },
      retrievalConfiguration: {
        vectorSearchConfiguration: { numberOfResults: 3 },
      },
    });

    const response = await bedrockClient.send(command);
    const results = response.retrievalResults || [];

    return results
      .filter((r) => r.content?.text)
      .map((r) => r.content.text)
      .join("\n\n---\n\n");
  } catch (error) {
    console.error("KB retrieval error:", error?.message || error);
    return "";
  }
}

export default async function handler(req, res) {
  // CORS headers
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const { messages } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: "Invalid messages format" });
    }

    // Get the latest user message for KB retrieval
    const lastUserMessage = [...messages]
      .reverse()
      .find((m) => m.role === "user");
    const query = lastUserMessage?.content || "";

    // Retrieve relevant knowledge base chunks
    const knowledgeContext = query
      ? await retrieveFromKnowledgeBase(query)
      : "";

    // Build system prompt with knowledge context
    let systemPrompt = SYSTEM_PROMPT;
    if (knowledgeContext) {
      systemPrompt += `\n\n## Company Knowledge\nUse the following information to answer the user's questions accurately. If the information doesn't cover their question, rely on your general guidelines above.\n\n${knowledgeContext}`;
    }

    const response = await client.messages.create({
      model: "claude-sonnet-4-5-20250929",
      max_tokens: 512,
      system: systemPrompt,
      messages: messages,
    });

    const reply = response.content[0].text;
    return res.status(200).json({ reply });
  } catch (error) {
    console.error("Error calling Claude API:", error?.message || error);
    return res.status(500).json({ error: "Failed to get response", detail: error?.message });
  }
}
