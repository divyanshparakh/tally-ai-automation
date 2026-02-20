"""
Unified LLM Wrapper
Supports Gemini, OpenRouter, and Ollama with a consistent interface
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Try to import all LLMs
try:
    import google.genai as genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

import requests

# ============================================================================
# ABSTRACT BASE CLASS FOR LLM CLIENTS
# ============================================================================

class LLMClientBase(ABC):
    """Abstract base class for all LLM clients"""
    
    @abstractmethod
    def chat(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        """
        Send a message and get a response.
        
        Args:
            prompt: User prompt
            system: System instruction (optional)
            temperature: Temperature setting (0.0 - 1.0)
            
        Returns:
            AI response text
        """
        pass


# ============================================================================
# GEMINI CLIENT
# ============================================================================

class GeminiLLMClient(LLMClientBase):
    """Google Gemini API client"""
    
    def __init__(self, api_key: str):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google Gemini API key
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai package not installed")
        
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash-lite"
        logger.info(f"✅ Gemini client initialized (model: {self.model})")
    
    def chat(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        """Send a message using Gemini API"""
        try:
            chat = self.client.chats.create(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                )
            )
            response = chat.send_message(prompt)
            return response.text.strip() if hasattr(response, 'text') else str(response).strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return ""


# ============================================================================
# OPENROUTER CLIENT
# ============================================================================

class OpenRouterLLMClient(LLMClientBase):
    """OpenRouter API client (via OpenAI SDK)"""
    
    def __init__(self, api_key: str, model: str = "meta-llama/llama-2-70b-chat"):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
            model: Model to use (default: Llama 2 70B)
        """
        if not OPENROUTER_AVAILABLE:
            raise ImportError("openai package not installed")
        
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={"HTTP-Referer": "http://localhost:5000"}
        )
        logger.info(f"✅ OpenRouter client initialized (model: {self.model})")
    
    def chat(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        """Send a message using OpenRouter API"""
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=2048
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            return ""


# ============================================================================
# OLLAMA CLIENT
# ============================================================================

class OllamaLLMClient(LLMClientBase):
    """Local Ollama LLM client"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", 
                 model: str = "gemma:7b", timeout: float = 5000):
        """
        Initialize Ollama client.
        
        Args:
            ollama_url: Ollama server URL
            model: Model name (e.g., gemma:7b, llama2)
            timeout: Request timeout in milliseconds
        """
        self.ollama_url = ollama_url
        self.model = model
        self.timeout = timeout / 1000.0  # Convert milliseconds to seconds
        logger.info(f"✅ Ollama client initialized (model: {self.model}, url: {ollama_url})")
    
    def chat(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        """Send a message using Ollama"""
        try:
            # Combine system and user prompt
            full_prompt = prompt
            if system:
                full_prompt = f"{system}\n\n{prompt}"
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "temperature": temperature,
                "stream": False
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            return data.get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return ""


# ============================================================================
# UNIFIED LLM CLIENT FACTORY
# ============================================================================

class LLMClientFactory:
    """Factory for creating LLM clients based on configuration"""
    
    @staticmethod
    def create_client(model_to_use: str = None) -> Optional[LLMClientBase]:
        """
        Create an LLM client based on environment configuration.
        
        Args:
            model_to_use: Model selection ("GEMINI", "OPENROUTER", "OLLAMA")
                         If None, uses MODEL_TO_USE env var
        
        Returns:
            Initialized LLM client or None if not available
        """
        model_to_use = model_to_use or os.getenv("MODEL_TO_USE", "GEMINI")
        
        if model_to_use == "GEMINI":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("❌ GEMINI selected but API_KEY not set in environment")
                return None
            
            if not GEMINI_AVAILABLE:
                logger.error("❌ GEMINI selected but google-genai not installed")
                return None
            
            try:
                return GeminiLLMClient(api_key)
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini: {e}")
                return None
        
        elif model_to_use == "OPENROUTER":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                logger.error("❌ OPENROUTER selected but OPENROUTER_API_KEY not set in environment")
                return None
            
            if not OPENROUTER_AVAILABLE:
                logger.error("❌ OPENROUTER selected but openai package not installed")
                return None
            
            try:
                model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
                return OpenRouterLLMClient(api_key, model)
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenRouter: {e}")
                return None
        
        elif model_to_use == "OLLAMA":
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
            model = os.getenv("GEMMA_MODEL", "gemma:7b")
            timeout = float(os.getenv("OLLAMA_HTTP_TIMEOUT", "5000"))
            
            try:
                return OllamaLLMClient(ollama_url, model, timeout)
            except Exception as e:
                logger.error(f"❌ Failed to initialize Ollama: {e}")
                return None
        
        else:
            logger.error(f"❌ Unknown MODEL_TO_USE: {model_to_use}")
            return None


# ============================================================================
# BACKWARDS COMPATIBILITY WRAPPER
# ============================================================================

class LegacyLLMClientWrapper:
    """
    Wrapper to provide legacy interface compatibility.
    This allows existing code using self.ai_client.chats.create() to work.
    """
    
    def __init__(self, unified_client: LLMClientBase):
        """
        Initialize wrapper with a unified LLM client.
        
        Args:
            unified_client: An instance of LLMClientBase
        """
        self.unified_client = unified_client
        self.chats = self._ChatsProxy(unified_client)
    
    class _ChatsProxy:
        """Proxy to provide chats.create() interface"""
        
        def __init__(self, unified_client: LLMClientBase):
            self.unified_client = unified_client
        
        def create(self, model: str = None, config=None):
            """Create a chat session"""
            return LegacyLLMClientWrapper._ChatSession(self.unified_client, config)
    
    class _ChatSession:
        """Represents a chat session"""
        
        def __init__(self, unified_client: LLMClientBase, config=None):
            self.unified_client = unified_client
            self.temperature = 0.7
            self.system_instruction = None
            
            if config and hasattr(config, 'temperature'):
                self.temperature = config.temperature
            if config and hasattr(config, 'system_instruction'):
                self.system_instruction = config.system_instruction
        
        def send_message(self, prompt: str):
            """Send a message and get response"""
            response_text = self.unified_client.chat(
                prompt,
                system=self.system_instruction,
                temperature=self.temperature
            )
            return LegacyLLMClientWrapper._Response(response_text)
    
    class _Response:
        """Response wrapper"""
        
        def __init__(self, text: str):
            self.text = text


def get_llm_client(model_to_use: str = None) -> Optional[LLMClientBase]:
    """
    Get an LLM client instance.
    
    Args:
        model_to_use: Model selection ("GEMINI", "OPENROUTER", "OLLAMA")
    
    Returns:
        Initialized LLM client or None
    """
    return LLMClientFactory.create_client(model_to_use)


def get_legacy_compatible_client(model_to_use: str = None) -> Optional[LegacyLLMClientWrapper]:
    """
    Get a legacy-compatible LLM client wrapper.
    Use this for existing code that expects the Gemini API interface.
    
    Args:
        model_to_use: Model selection ("GEMINI", "OPENROUTER", "OLLAMA")
    
    Returns:
        Wrapped LLM client or None
    """
    client = LLMClientFactory.create_client(model_to_use)
    if client:
        return LegacyLLMClientWrapper(client)
    return None
