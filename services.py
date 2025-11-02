# services.py
import logging
from openai import AzureOpenAI, OpenAIError
from config import Settings
from schemas import ChatRequest

# Set up a logger
logger = logging.getLogger(__name__)

# This will store chat histories.
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
            logger.info("AzureOpenAI client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize AzureOpenAI client: {e}")
            self.client = None

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

    def _mock_rag_retriever(self, intent: str) -> str:
        """
        [MOCK] Simulates a RAG query to a vector database.
        """
        logger.info(f"Mock RAG retrieving for intent: {intent}")
        rag_db = {
            "work_stress": "Retrieved technique: 'The 5-4-3-2-1 grounding technique. Focus on five things you see, four things you feel, three things you hear, two things you smell, and one thing you taste. This helps anchor you to the present moment.'",
            "study_anxiety": "Retrieved technique: 'The Pomodoro Technique. Break your study time into 25-minute focused intervals, followed by a 5-minute break. This makes the task less daunting.'",
            "feeling_depressed": "Retrieved affirmation: 'It's okay to not be okay. Your feelings are valid. Remind the user to be kind to themselves and that this feeling will pass.'",
            "general_greeting": "Retrieved context: 'User is starting the conversation. Be warm and welcoming. Ask an open-ended question about how they are feeling today.'",
            "affirm": "Retrieved context: 'The user has agreed to a suggestion. Be encouraging and proceed with the next step.'",
            "deny": "Retrieved context: 'The user has declined a suggestion. Be gentle, validate their choice, and offer an alternative, like just talking.'"
        }
        return rag_db.get(intent, "Retrieved context: 'Acknowledge the user's statement and ask a gentle, clarifying question.'")

    def _build_llm_prompt(self, message: str, intent: str, context: str, history: list[dict]) -> list[dict]:
        """
        Builds the structured prompt for the LLM, now including history.
        """
        
        # --- !! NAME CHANGE IS HERE !! ---
        SYSTEM_PROMPT = (
            "You are 'Luma', an AI emotional support chatbot. Your role is to "
            "help working professionals and students navigate stress and negative emotions. "
            "You are empathetic, patient, and non-judgmental. "
            "NEVER give medical advice. "
            "Use the 'Retrieved Context' to help guide your response. "
            "The 'User Intent' is for your information. Do not mention it explicitly. "
            "Keep your responses concise, supportive, and end with a question "
            "to encourage the user to keep talking."
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({
            "role": "user",
            "content": (
                f"User Message: \"{message}\"\n\n"
                f"--- (Internal analysis) ---\n"
                f"User Intent: {intent}\n"
                f"Retrieved Context: {context}\n"
                f"--- (End analysis) ---\n\n"
                f"Your response:"
            )
        })
        
        return messages

    def get_chat_response(self, request: ChatRequest) -> tuple[str, str, str]:
        """
        Main orchestration function.
        """
        if not self.client:
            logger.error("LLM client is not initialized. Cannot process request.")
            raise OpenAIError("The AI service is not configured correctly.")
            
        session_id = request.session_id
        if not session_id:
            logger.warning("No session_id provided. Using 'default' session.")
            session_id = "default_session"

        # 1. Get history from cache
        history = CHAT_HISTORY_CACHE.get(session_id, [])

        # 2. Rasa NLU Step
        nlu_data = self._mock_rasa_nlu(request.message)
        intent = nlu_data.get("intent", {}).get("name", "unknown")

        # 3. RAG Step
        context = self._mock_rag_retriever(intent)
        
        logger.info(f"[Session: {session_id}] Retrieved RAG context for intent '{intent}': {context}")

        # 4. LLM Generation Step
        prompt = self._build_llm_prompt(request.message, intent, context, history)
        
        logger.info(f"[Session: {session_id}] Sending full prompt to LLM: {prompt}")

        try:
            completion = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=prompt,
                temperature=0.7,
                max_tokens=200,
            )
            
            llm_response = completion.choices[0].message.content.strip()
            
            # 5. Save to cache
            history.append({"role": "user", "content": request.message})
            history.append({"role": "assistant", "content": llm_response})
            CHAT_HISTORY_CACHE[session_id] = history

            return llm_response, intent, context

        except OpenAIError as e:
            logger.error(f"Error calling Azure OpenAI: {e}")
            raise