from app.tally_core.interface import TallyInterface
from app.tally_agent.llm_wrapper import get_llm_client, get_legacy_compatible_client
import os

# Test Tally Interface
tallyInterface = TallyInterface(os.getenv("TALLY_HOST", "http://localhost"), int(os.getenv("TALLY_PORT", "9000")))
print(tallyInterface.get_all_ledgers_list())
print(tallyInterface.get_all_stock_items())
print(tallyInterface.get_current_company_detail())

# Test LLM Clients
print("\n=== Testing LLM Clients ===\n")

# Test Gemini
print("Testing Gemini LLM Client:")
gemini_client = get_llm_client("GEMINI")
if gemini_client:
    response = gemini_client.chat("Hello, what is 2+2?")
    print(f"Gemini Response: {response}\n")
else:
    print("Gemini client not available\n")

# Test OpenRouter
print("Testing OpenRouter LLM Client:")
openrouter_client = get_llm_client("OPENROUTER")
if openrouter_client:
    response = openrouter_client.chat("Hello, what is 2+2?")
    print(f"OpenRouter Response: {response}\n")
else:
    print("OpenRouter client not available\n")

# Test Ollama
print("Testing Ollama LLM Client:")
ollama_client = get_llm_client("OLLAMA")
if ollama_client:
    response = ollama_client.chat("Hello, what is 2+2?")
    print(f"Ollama Response: {response}\n")
else:
    print("Ollama client not available\n")

# Test Legacy Compatible Client
print("Testing Legacy Compatible Client:")
legacy_client = get_legacy_compatible_client()
if legacy_client:
    chat_session = legacy_client.chats.create()
    response = chat_session.send_message("Hello, what is 2+2?")
    print(f"Legacy Client Response: {response.text}\n")
else:
    print("Legacy compatible client not available\n")
