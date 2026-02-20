from app.tally_core.interface import TallyInterface
from app.tally_agent.llm_wrapper import get_llm_client, get_legacy_compatible_client
from app.tally_agent.orchestrator import OrchestratorAgent
from app.tally_agent.sales.sales_voucher_agent import SalesVoucherAgent
import os

if __name__ == "__main__":
    tallyInterface = TallyInterface(
        os.getenv("TALLY_HOST", "http://localhost"),
        int(os.getenv("TALLY_PORT", "9000"))
    )
    llm_client = get_llm_client()

    agents = {
        "create_sales_voucher": SalesVoucherAgent(tallyInterface, llm_client),
        # Add more agents here as needed
    }
    orchestrator = OrchestratorAgent(llm_client, agents)
    user_input = input("Enter your request: ")
    result = orchestrator.run(user_input)
    print(f"Orchestrator result: {result}")
