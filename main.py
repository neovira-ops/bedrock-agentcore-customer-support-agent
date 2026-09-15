"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the step-by-step solution files and INSTRUCTIONS.md for guidance.
Do NOT copy the solution directly — work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
# Create a BedrockAgentCoreApp instance.
# This registers the ASGI server for AgentCore deployment.
# There must be exactly one instance per deployment.
#
# Hint: app = BedrockAgentCoreApp()

# TODO: Create the BedrockAgentCoreApp instance
app = BedrockAgentCoreApp()  


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
# Replace the placeholder strings with your actual AWS resource values.
# You collected these in Part 1 of the INSTRUCTIONS.
#
# GATEWAY_URL format: https://<alias>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
# KB_ID       format: 10-character alphanumeric string from the KB console
# REGION:     your AWS region, e.g. "us-east-1"
# MEMORY_ID   format: shown in the AgentCore Memory console

GATEWAY_URL = "https://customersupportgateway-jut41bnjwc.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID = "R2CSXBE0FP"
REGION = "us-east-1"
MEMORY_ID = "CustomerSupportMemory-NDYTb6AGbB"


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
# Create:
#   1. A BedrockModel using model_id "global.amazon.nova-2-lite-v1:0"
#   2. A MemoryClient with region_name=REGION
#   3. A boto3 client for the "bedrock-agent-runtime" service in REGION
#
# Hint: model = BedrockModel(model_id=model_id)

model_id = "global.amazon.nova-2-lite-v1:0"

# Bedrock model used by the Strands agent
model = BedrockModel(
    model_id="global.amazon.nova-2-lite-v1:0",
    region_name=REGION,
)

# AgentCore Memory client
memory_client = MemoryClient(region_name=REGION)

# Bedrock Agent Runtime client used for Knowledge Base retrieval
bedrock_agent_runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION,
)


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
# Implement get_namespaces() to return a dict mapping strategy type to
# namespace template string.
#
# Steps:
#   1. Call mem_client.get_memory_strategies(memory_id) to get strategy list
#   2. Return a dict: { strategy["type"]: strategy["namespaces"][0] for each strategy }
#
# Example output:
#   { "SEMANTIC": "cs_agent/{actorId}/facts",
#     "USER_PREFERENCE": "cs_agent/{actorId}/preferences" }

def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return memory strategy information keyed by strategy type."""
    strategies = mem_client.get_memory_strategies(memory_id)

    namespaces = {}

    for strategy in strategies:
        strategy_type = strategy["type"]
        strategy_id = strategy.get("strategyId")

        if "namespaceTemplates" in strategy:
            namespace_template = strategy["namespaceTemplates"][0]
        elif "namespaces" in strategy:
            namespace_template = strategy["namespaces"][0]
        else:
            continue

        namespaces[strategy_type] = {
            "strategy_id": strategy_id,
            "namespace_template": namespace_template,
        }

    return namespaces



# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
# Implement MemoryHook, a HookProvider subclass that adds long-term memory.
#
# The class needs:
#   __init__(self, actor_id, session_id, memory_client, memory_id)
#     — store all four as instance attributes
#     — call get_namespaces() and store the result as self.namespaces
#
#   retrieve_customer_context(self, event: MessageAddedEvent)
#     — only runs for plain-text user messages (not tool results)
#     — for each strategy namespace, call memory_client.retrieve_memories(
#          memory_id, namespace (formatted with actorId), query, top_k=5)
#     — collect non-empty memory texts tagged with their strategy type
#     — if any memories found, prepend them to the user message as:
#          "Customer Context:\n<memories>\n\n<original_message>"
#
#   save_support_interaction(self, event: AfterInvocationEvent)
#     — walk the message list backwards to find the last plain-text user
#       query and the last assistant response
#     — call memory_client.create_event(memory_id, actor_id, session_id,
#          messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")])
#
#   register_hooks(self, registry: HookRegistry)
#     — register retrieve_customer_context on MessageAddedEvent
#     — register save_support_interaction on AfterInvocationEvent

class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id

        self.namespaces = get_namespaces(
            memory_client,
            memory_id,
        )

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""

        messages = event.agent.messages

        if not messages:
            return

        # Get the most recent message.
        message = messages[-1]

        # Only process user messages.
        if message.get("role") != "user":
            return

        content = message.get("content", [])

        # Only process plain-text messages.
        if (
            not isinstance(content, list)
            or not content
            or not isinstance(content[0], dict)
            or "text" not in content[0]
        ):
            return

        query = content[0]["text"].strip()

        if not query:
            return

        memories_found = []

        for strategy_type, strategy_info in self.namespaces.items():
            namespace_template = strategy_info["namespace_template"]
            strategy_id = strategy_info["strategy_id"]

            namespace = namespace_template.format(
                actorId=self.actor_id,
                sessionId=self.session_id,
                memoryStrategyId=strategy_id,
            )


            try:
                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=query,
                    top_k=5,
                )

                for memory in memories:
                    memory_content = memory.get("content", {})

                    if isinstance(memory_content, dict):
                        memory_text = memory_content.get("text", "")
                    else:
                        memory_text = ""

                    if memory_text and memory_text.strip():
                        memories_found.append(
                            f"[{strategy_type}] {memory_text.strip()}"
                        )

            except Exception as exc:
                logger.warning(
                    "Failed to retrieve %s memories: %s",
                    strategy_type,
                    exc,
                )

        if memories_found:
            customer_context = "\n".join(memories_found)

            message["content"] = [
                {
                    "text": (
                        f"Customer Context:\n"
                        f"{customer_context}\n\n"
                        f"{query}"
                    )
                }
            ]
        else:
            return

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""

        messages = event.agent.messages

        customer_query = None
        agent_response = None

        # Walk backwards through the messages.
        for message in reversed(messages):
            role = message.get("role")
            content = message.get("content", [])

            # Only consider plain-text messages.
            if (
                not isinstance(content, list)
                or not content
                or not isinstance(content[0], dict)
                or "text" not in content[0]
            ):
                continue

            text = content[0]["text"].strip()

            if not text:
                continue

            if customer_query is None and role == "user":
                customer_query = text

            elif agent_response is None and role == "assistant":
                agent_response = text

            if customer_query and agent_response:
                break

        if customer_query and agent_response:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[
                    (customer_query, "USER"),
                    (agent_response, "ASSISTANT"),
                ],
            )

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""

        registry.add_callback(
            MessageAddedEvent,
            self.retrieve_customer_context,
        )

        registry.add_callback(
            AfterInvocationEvent,
            self.save_support_interaction,
        )


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
# Implement search_knowledge_base(query) using the @tool decorator.
#
# Steps:
#   1. Guard: if KB_ID is empty return "Knowledge base not configured."
#   2. Call _bedrock_runtime.retrieve(
#          knowledgeBaseId=KB_ID,
#          retrievalQuery={"text": query}
#      )
#   3. Extract resp["retrievalResults"]; return a message if empty
#   4. Join the text chunks with "\n---\n" and return the result
#
# The docstring is the tool description — the model uses it to decide when
# to call this tool, so keep it clear and accurate.

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    # TODO: Implement the Knowledge Base search
    if not KB_ID:
        return "Knowledge base not configured."

    resp = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": query},
    )

    results = resp.get("retrievalResults", [])

    if not results:
        return "No relevant information found in the knowledge base."

    chunks = []

    for result in results:
        text = result.get("content", {}).get("text", "")

        if text:
            chunks.append(text)

    if not chunks:
        return "No relevant information found in the knowledge base."

    return "\n---\n".join(chunks)


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
# Implement calculate_loyalty_discount() using the @tool decorator.
#
# The tool must:
#   1. Build a self-contained Python code string that:
#        • Defines earn_rates: {"standard": 1, "device": 2, "fresh": 5}
#        • Defines tier_rates: {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
#        • Calculates points_redeemed (floor to nearest 500, cap at 50% of order)
#        • Calculates tier_discount (applied to subtotal after points)
#        • Calculates final_total, total_savings, points_earned, remaining_points
#        • Prints a JSON result dict
#   2. Execute the code with code_session(REGION).invoke("executeCode", {...})
#      using language="python" and clearContext=True
#   3. Return the first result event as a JSON string
#   4. Include a fallback that computes only the tier discount if the
#      Code Interpreter is unavailable

@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    # Build a self-contained program for the AgentCore Code Interpreter.
    code = f"""
import json
import math

earn_rates = {{
    "standard": 1,
    "device": 2,
    "fresh": 5,
}}

tier_rates = {{
    "Silver": 0.00,
    "Gold": 0.10,
    "Platinum": 0.15,
}}

loyalty_points = {loyalty_points}
tier = {tier!r}
order_total = {order_total}
product_category = {product_category!r}

# Loyalty points are worth $0.01 each.
# Points must be redeemed in blocks of 500.
available_redemption_blocks = loyalty_points // 500

# Points can cover at most 50% of the order.
max_redemption_points = math.floor(
    (order_total * 0.50 * 100) / 500
) * 500

points_redeemed = min(
    available_redemption_blocks * 500,
    max_redemption_points,
)

points_discount = points_redeemed / 100

# Apply the tier discount after the points discount.
subtotal_after_points = max(
    0.0,
    order_total - points_discount,
)

tier_discount_pct = tier_rates.get(tier, 0.00)

tier_discount = (
    subtotal_after_points * tier_discount_pct
)

final_total = max(
    0.0,
    subtotal_after_points - tier_discount,
)

total_savings = (
    order_total - final_total
)

points_earned = math.floor(
    order_total * earn_rates.get(product_category, 1)
)

remaining_points = (
    loyalty_points
    - points_redeemed
    + points_earned
)

result = {{
    "points_redeemed": points_redeemed,
    "tier_discount_pct": tier_discount_pct,
    "final_total": round(final_total, 2),
    "remaining_points": remaining_points,
    "points_earned": points_earned,
    "total_savings": round(total_savings, 2),
}}

print(json.dumps(result))
"""

    try:
        with code_session(REGION) as code_client:
            response = code_client.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True,
                },
            )

        for event in response["stream"]:
            return json.dumps(event["result"])

        return json.dumps({
            "error": "Code Interpreter returned no result."
        })

    except Exception as e:
        # Fallback: calculate only the tier discount locally.
        tier_rates = {
            "Silver": 0.00,
            "Gold": 0.10,
            "Platinum": 0.15,
        }

        tier_discount_pct = tier_rates.get(tier, 0.00)
        tier_discount = order_total * tier_discount_pct
        final_total = order_total - tier_discount

        return json.dumps({
            "points_redeemed": 0,
            "tier_discount_pct": tier_discount_pct,
            "final_total": round(final_total, 2),
            "remaining_points": loyalty_points,
            "points_earned": 0,
            "total_savings": round(tier_discount, 2),
            "fallback": True,
            "error": str(e),
        })


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
# Implement the invoke() function decorated with @app.entrypoint.
#
# Steps:
#   1. Extract user_input, actor_id, and session_id from the payload
#      (generate a UUID if session_id is missing)
#   2. Instantiate MemoryHook for this actor/session
#   3. Instantiate AgentCoreBrowser(region=REGION)
#   4. Build the tools list: [search_knowledge_base, calculate_loyalty_discount,
#                              agent_core_browser.browser]
#   5. Connect to the Gateway via MCPClient, load gateway_tools, extend tools list
#   6. Create and invoke the Agent with all tools, hooks, and system_prompt
#   7. Return the text from the first content block of the response
#   8. Handle exceptions gracefully

# TODO 8: Implement the AgentCore entrypoint

@app.entrypoint
async def invoke(payload, context=None):
    """
    Main AgentCore entrypoint.

    Connects to the AgentCore Gateway, loads its tools, configures
    the customer memory hook and invokes the Strands agent.
    """
    try:
        # Get the user's message
        user_input = payload.get("prompt", "")

        if not user_input:
            return "Please provide a prompt."

        # Get customer/session information
        actor_id = payload.get("customer_id", "anonymous")
        session_id = payload.get("session_id", str(uuid.uuid4()))

        # Create the memory hook
        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        # Create the AgentCore Browser
        agent_core_browser = AgentCoreBrowser(region=REGION)

        # Connect to the AgentCore Gateway using IAM/SigV4 authentication
        mcp_client = MCPClient(
            lambda: aws_iam_streamablehttp_client(
                endpoint=GATEWAY_URL,
                aws_region=REGION,
                aws_service="bedrock-agentcore",
            )
        )

        # Start the MCP connection and load Gateway tools
        with mcp_client:
            gateway_tools = mcp_client.list_tools_sync()

            # Build the complete tool list
            tools = [
                search_knowledge_base,
                calculate_loyalty_discount,
                agent_core_browser.browser,
            ]

            tools.extend(gateway_tools)

            # Create the Strands agent
            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=(
                    "You are a helpful customer support agent. "
                    "Use the available tools to help customers with "
                    "orders, refunds, product information, loyalty "
                    "discounts, and other support requests. "
                    "Use the knowledge base when product information "
                    "is needed. Use Gateway tools for customer and "
                    "order information or refund operations. "
                    "Use the browser when live webpage information "
                    "is required."
                    "When using the browser tool, always use a "
                    "session_name containing only lowercase letters, "
                    "numbers, and hyphens, between 10 and 36 characters "
                    "long. Use names such as 'browser-session-1'."
                ),
            )

            # Invoke the agent
            response = await agent.invoke_async(user_input)

            # Return the agent's response text
            return response.message["content"][0]["text"]

    except Exception as e:
        logger.exception("Agent invocation failed")
        return (
            "Sorry, I encountered an error while processing "
            f"your request: {e}"
        )


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    main()
