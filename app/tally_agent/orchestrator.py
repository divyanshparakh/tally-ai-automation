"""
OrchestratorAgent: Uses LLM to decide which agent to invoke for a user query.
"""
from typing import Dict, Any

class OrchestratorAgent:
    def __init__(self, llm_client, agents: Dict[str, Any]):
        """
        llm_client: An LLM client with a .chat() method
        agents: Dict mapping agent names to agent instances (must have .run(user_input) method)
        """
        self.llm_client = llm_client
        self.agents = agents

    def decide_agent(self, user_input: str) -> str:
        """
        Use LLM to decide which agent to invoke. Returns agent name key.
        Returns the exact agent key from self.agents, or 'none'.
        """
        agent_keys = ", ".join(self.agents.keys())
        system_prompt = (
            "You are a Tally ERP request router.\n"
            f"Available agents (respond with EXACTLY one of these keys): {agent_keys}\n\n"
            "- User wants to create/add/record a sales invoice or sales voucher → create_sales_voucher\n"  # ← CHANGED: exact key
            "- No suitable agent found → none\n\n"
            "Respond with ONLY the key, nothing else."
        )
        response = self.llm_client.chat(user_input, system=system_prompt, temperature=0.0)
        agent_key = response.strip().lower()
        return agent_key if agent_key in self.agents else "none"  # ← CHANGED: safe fallback

    def run(self, user_input: str, context: dict = None) -> Any:
        agent_key = self.decide_agent(user_input)
        if agent_key in self.agents:
            return self.agents[agent_key].run(user_input, context)
        return {"status": "no_agent", "message": "No suitable agent found. Detected intent: '{agent_key}'"}
