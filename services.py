# services.py
import logging
from openai import AzureOpenAI, OpenAIError
from config import Settings
from schemas import ChatRequest

# ✅ NEW IMPORTS for your custom NLU & RAG logic
from nlu.intent_emotion import analyze_text
from rag.retriever import retrieve_relevant

# Set up a logger
logger = logging.getLogger(__name__)

# This will store chat histories (simple in-memory cache)
CHAT_HISTORY_CACHE = {}


class ChatService:
    """
    Manages the conversation logic, integrating NLU, RAG, and LLM.
    """
    def __init__(self, settings: Settings):
        """
        Initializes the AzureOpenAI client upon creation.
        """
        try:
            self.client = AzureOpenAI(
                api_key=settings.AZURE_OPENAI_KEY,
                api_version=settings.AZURE_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            )
            self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME
            logger.info("✅ AzureOpenAI client initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize AzureOpenAI client: {e}")
            self.client = None

    # ---------------------------------------------
    # 🔹 Optional mock NLU (kept for fallback use)
    # ---------------------------------------------
    def _mock_rasa_nlu(self, message: str) -> dict:
        """
        [MOCK] Simulates a call to a Rasa NLU server.
        """
        logger.info(f"Mock NLU processing message: {message}")
        message_lower = message.lower()

        if "yes" in message_lower or "ok" in message_lower or "sure" in message_lower:
            return {"intent": {"name": "affirm"}, "entities": []}
        if "no" in message_lower or "not really" in message_lower:
            return {"intent": {"name": "deny"}, "entities": []}
        if "job" in message_lower or "deadline" in message_lower or "work" in message_lower:
            return {"intent": {"name": "work_stress"}, "entities": []}
        if "exam" in message_lower or "study" in message_lower:
            return {"intent": {"name": "study_anxiety"}, "entities": []}
        if "sad" in message_lower or "lonely" in message_lower:
            return {"intent": {"name": "feeling_depressed"}, "entities": []}

        return {"intent": {"name": "general_greeting"}, "entities": []}

    # ---------------------------------------------
    # 🔹 Optional mock RAG retriever (fallback)
    # ---------------------------------------------
    def _mock_rag_retriever(self, intent: str) -> str:
        """
        [MOCK] Simulates a RAG query to a vector database.
        """
        logger.info(f"Mock RAG retrieving for intent: {intent}")
        rag_db = {
            "work_stress": "Retrieved technique: 'Try the 5-4-3-2-1 grounding technique to relax.'",
            "study_anxiety": "Retrieved tip: 'Use the Pomodoro Technique — 25 min study + 5 min break.'",
            "feeling_depressed": "Retrieved affirmation: 'It's okay to not be okay. Be kind to yourself.'",
            "general_greeting": "Retrieved context: 'Be warm and welcoming. Ask how they’re feeling today.'",
            "affirm": "Retrieved context: 'The user agreed. Continue positively.'",
            "deny": "Retrieved context: 'The user declined. Offer gentle alternatives.'"
        }
        return rag_db.get(intent, "Retrieved context: 'Acknowledge and respond empathetically.'")

    # ---------------------------------------------
    # 🔹 Build the LLM prompt (with context + history)
    # ---------------------------------------------
    def _build_llm_prompt(self, message: str, intent: str, context: str, history: list[dict]) -> list[dict]:
        """
        Builds the structured prompt for the LLM, including conversation history.
        """
        SYSTEM_PROMPT = (
            "You are 'Luma', an empathetic AI emotional support chatbot for students and professionals. "
            "Help users express and regulate emotions. "
            "NEVER give medical advice. "
            "Use the retrieved context and detected intent internally to guide your tone. "
            "End your responses with an open-ended question to keep the user engaged."
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({
            "role": "user",
            "content": (
                f"User Message: \"{message}\"\n\n"
                f"--- (Internal info) ---\n"
                f"User Intent: {intent}\n"
                f"Retrieved Context: {context}\n"
                f"--- (End info) ---\n\n"
                f"Your response:"
            )
        })
        return messages

    # ---------------------------------------------
    # 🔹 Main function: generates chatbot response
    # ---------------------------------------------
    def get_chat_response(self, request: ChatRequest) -> tuple[str, str, str]:
        """
        Main orchestration: NLU → RAG → LLM → Response
        """
        if not self.client:
            logger.error("❌ LLM client not initialized.")
            raise OpenAIError("The AI service is not configured correctly.")

        session_id = request.session_id or "default_session"

        # 1️⃣ Load chat history
        history = CHAT_HISTORY_CACHE.get(session_id, [])

        # 2️⃣ NEW — NLU step using your custom module
        try:
            nlu_result = analyze_text(request.message)
            intent = nlu_result.get("intent", "unknown")
            emotion = nlu_result.get("emotion", "neutral")
            logger.info(f"[Session: {session_id}] NLU detected intent='{intent}', emotion='{emotion}'")
        except Exception as e:
            logger.warning(f"⚠️ Custom NLU failed: {e}, falling back to mock.")
            nlu_data = self._mock_rasa_nlu(request.message)
            intent = nlu_data.get("intent", {}).get("name", "unknown")
            emotion = "neutral"

        # 3️⃣ NEW — RAG step using your retriever
        try:
            retrieved_docs = retrieve_relevant(request.message)
            context = retrieved_docs[0] if retrieved_docs else self._mock_rag_retriever(intent)
        except Exception as e:
            logger.warning(f"⚠️ RAG retriever failed: {e}, using mock retriever.")
            context = self._mock_rag_retriever(intent)

        logger.info(f"[Session: {session_id}] RAG context: {context}")

        # 4️⃣ Build LLM prompt
        prompt = self._build_llm_prompt(request.message, intent, context, history)

        # 5️⃣ Generate LLM response
        try:
            completion = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=prompt,
                temperature=0.7,
                max_tokens=250,
            )
            llm_response = completion.choices[0].message.content.strip()

            # 6️⃣ Update conversation history
            history.append({"role": "user", "content": request.message})
            history.append({"role": "assistant", "content": llm_response})
            CHAT_HISTORY_CACHE[session_id] = history

            logger.info(f"[Session: {session_id}] Response generated successfully.")
            return llm_response, intent, context

        except OpenAIError as e:
            logger.error(f"❌ Error calling Azure OpenAI: {e}")
            raise
