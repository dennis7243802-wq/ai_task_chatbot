import streamlit as st
from supabase import create_client, Client
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Optional

# ==========================================
# 1. Page Configuration & Client Initialization
# ==========================================
st.set_page_config(
    page_title="AI Task Organizer Prototype",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def init_supabase() -> Client:
    """Initialize and cache the Supabase client using Streamlit secrets."""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

def get_available_api_keys() -> List[str]:
    """Retrieve up to 5 Gemini API keys from Streamlit secrets."""
    keys = []
    for i in range(1, 6):
        key_name = f"GEMINI_API_KEY_{i}"
        if key_name in st.secrets and st.secrets[key_name]:
            keys.append(st.secrets[key_name])
    # Fallback if someone only defined GEMINI_API_KEY
    if not keys and "GEMINI_API_KEY" in st.secrets:
        keys.append(st.secrets["GEMINI_API_KEY"])
    return keys

# ==========================================
# 2. Robust Gemini Client with Key Rotation & Model Fallback
# ==========================================
def generate_with_key_rotation(
    contents: Any,
    config: Optional[types.GenerateContentConfig] = None
) -> str:
    """
    Rotates through up to 5 API keys. For each key, attempts to use
    'gemini-3.5-flash'. If that model fails, falls back to 'gemini-3.8-flash'.
    If all attempts fail, raises an exception.
    """
api_keys = get_available_api_keys()
    if not api_keys:
        raise ValueError("No Gemini API keys found in Streamlit secrets!")

    models_to_try = ["gemini-3.5-flash", "gemini-3.8-flash"]
    
    last_exception = None

    for key_idx, api_key in enumerate(api_keys, start=1):
        try:
            client = genai.Client(api_key=api_key)
            
            for model_name in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    # Optional: Display a subtle info toast or log successful routing if needed
                    return response.text
                except Exception as model_err:
                    last_exception = model_err
                    # Model failed with current key, try next model or next key
                    continue
                    
        except Exception as key_err:
            last_exception = key_err
            # Current key failed entirely, move to next key in rotation
            continue

    raise RuntimeError(
        f"All API keys and models exhausted. Last encountered error: {last_exception}"
    )

# ==========================================
# 3. Pydantic Schema for Agent Intent Routing
# ==========================================
class IntentClassification(BaseModel):
    intent: Literal[
        "translation",
        "summarization",
        "action_item_extraction",
        "email_drafting",
        "out_of_scope"
    ] = Field(
        description="The categorized intent of the user's request."
    )
    task_title: str = Field(
        description="A concise 3-6 word summary of the task intent (e.g., 'Translate Text to Spanish')."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 regarding the intent classification."
    )
    reasoning: str = Field(
        description="Clear, polite explanation to the user if confidence < 0.7 or if out_of_scope."
    )

# ==========================================
# 4. Supabase Database Operations
# ==========================================
def get_user_tasks(user_id: str) -> List[Dict[str, Any]]:
    response = (
        supabase.table("tasks")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return response.data

def create_new_task(user_id: str) -> Dict[str, Any]:
    task_res = (
        supabase.table("tasks")
        .insert({"user_id": user_id, "task_intent": "New Task Inquiry"})
        .execute()
    )
    new_task = task_res.data[0]
    task_id = new_task["task_id"]

    initial_greeting = (
        "Hello! I am your AI Task Assistant. How can I help organize or execute your work today?\n\n"
        "**Supported Workflows:**\n"
        "- 🌐 **Translation** (Translate text into any target language)\n"
        "- 📝 **Summarization** (Condense notes, articles, or transcripts)\n"
        "- ✅ **Action Item Extraction** (Turn messy notes into structured to-do lists)\n"
        "- ✉️ **Email Drafting** (Draft professional emails from bullet points)"
    )
    save_message(task_id, "assistant", initial_greeting)
    return new_task

def update_task_intent(task_id: str, task_intent: str) -> None:
    supabase.table("tasks").update({"task_intent": task_intent}).eq("task_id", task_id).execute()

def delete_task(task_id: str) -> None:
    supabase.table("tasks").delete().eq("task_id", task_id).execute()

def get_task_messages(task_id: str) -> List[Dict[str, Any]]:
    response = (
        supabase.table("messages")
        .select("*")
        .eq("task_id", task_id)
        .order("created_at", desc=False)
        .execute()
    )
    return response.data

def save_message(task_id: str, role: str, content: str) -> None:
    supabase.table("messages").insert({
        "task_id": task_id,
        "role": role,
        "content": content
    }).execute()

# ==========================================
# 5. AI Agent Workflow (Router + Executor)
# ==========================================
def classify_user_intent(user_message: str, chat_history: List[Dict[str, Any]]) -> IntentClassification:
    history_context = "\n".join(
        [f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history[-6:]]
    )

    system_instruction = (
        "You are an AI Intent Router for a task management assistant. "
        "Analyze the user's latest message alongside recent conversation context. "
        "Classify the request into one of the supported workflows: "
        "'translation', 'summarization', 'action_item_extraction', or 'email_drafting'. "
        "If the request does not fit these four workflows (e.g., coding, math, general chit-chat, weather), "
        "classify it as 'out_of_scope'. "
        "Assign a realistic confidence score (0.0 to 1.0). If the request is ambiguous or lacks source text "
        "to process, lower the confidence score below 0.7 and explain what details are needed in 'reasoning'."
    )

    prompt = f"Conversation Context:\n{history_context}\n\nLatest User Request:\n{user_message}"

    response_text = generate_with_key_rotation(
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=IntentClassification,
            temperature=0.1,
        )
    )
    return IntentClassification.model_validate_json(response_text)

def execute_task_workflow(intent: str, user_message: str, chat_history: List[Dict[str, Any]]) -> str:
    workflow_prompts = {
        "translation": (
            "You are an expert translator. Accurately translate the requested text while preserving "
            "tone, nuance, and formatting. Specify the target language clearly."
        ),
        "summarization": (
            "You are an executive assistant. Provide a concise, structured summary using bullet points "
            "highlighting key takeaways and core facts."
        ),
        "action_item_extraction": (
            "You are a productivity specialist. Extract all actionable tasks from the user's input. "
            "Format them as a clean checklist with suggested priority levels and owners if mentioned."
        ),
        "email_drafting": (
            "You are a professional communications specialist. Draft a clear, well-structured, and polite "
            "email based on the user's instructions, including a subject line."
        ),
    }

    system_instruction = workflow_prompts.get(intent, "You are a helpful task assistant.")
    history_context = "\n".join(
        [f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history[-6:]]
    )
    prompt = f"Context:\n{history_context}\n\nTask to Execute:\n{user_message}"

    return generate_with_key_rotation(
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.4,
        ),
    )

# ==========================================
# 6. Streamlit UI Layout & State Management
# ==========================================
if "active_task_id" not in st.session_state:
    st.session_state.active_task_id = None

# --- Sidebar: User Session & Task Management ---
with st.sidebar:
    st.title("🗂️ Task Workspace")
    st.caption("Phase 1 AI Agent Prototype (Multi-Key Rotation)")

    user_id = st.text_input("Active User ID", value="demo_user_001").strip()

    if not user_id:
        st.warning("Please enter a User ID to load tasks.")
        st.stop()

    st.divider()

    if st.button("➕ Create New Task", use_container_width=True, type="primary"):
        created_task = create_new_task(user_id)
        st.session_state.active_task_id = created_task["task_id"]
        st.rerun()

    st.subheader("Your Task Intents")
    user_tasks = get_user_tasks(user_id)

    if not user_tasks:
        created_task = create_new_task(user_id)
        st.session_state.active_task_id = created_task["task_id"]
        st.rerun()

    task_ids = [t["task_id"] for t in user_tasks]
    if st.session_state.active_task_id not in task_ids:
        st.session_state.active_task_id = task_ids[0]

    for task in user_tasks:
        t_id = task["task_id"]
        t_intent = task["task_intent"]
        is_selected = (t_id == st.session_state.active_task_id)

        col1, col2 = st.columns([0.82, 0.18])
        with col1:
            button_label = f"🟢 {t_intent}" if is_selected else f"⚪ {t_intent}"
            if st.button(
                button_label,
                key=f"select_{t_id}",
                use_container_width=True,
                help=f"Task ID: {t_id}"
            ):
                st.session_state.active_task_id = t_id
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{t_id}", help="Delete this task and its history"):
                delete_task(t_id)
                if st.session_state.active_task_id == t_id:
                    st.session_state.active_task_id = None
                st.rerun()

# --- Main Area: Active Task Conversation & Agent Execution ---
active_task = next(
    (t for t in user_tasks if t["task_id"] == st.session_state.active_task_id),
    None
)

if active_task:
    st.header(f"Task: {active_task['task_intent']}")
    st.caption(f"**User ID:** `{user_id}` | **Task ID:** `{active_task['task_id']}`")
    st.divider()

    messages = get_task_messages(active_task["task_id"])
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Describe the task you need help with..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        save_message(active_task["task_id"], "user", user_input)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Analyzing request intent with key rotation..."):
                    classification = classify_user_intent(user_input, messages)

                if classification.intent == "out_of_scope" or classification.confidence < 0.7:
                    fallback_response = (
                        f"⚠️ **Unable to Execute Task Automatically**\n\n"
                        f"- **Detected Category:** `{classification.intent}`\n"
                        f"- **Confidence Score:** `{classification.confidence:.2f}` (Threshold: `0.70`)\n\n"
                        f"**Reason:** {classification.reasoning}\n\n"
                        f"Please clarify your request or provide specific text/details for one of the supported workflows "
                        f"(*Translation, Summarization, Action Item Extraction, or Email Drafting*)."
                    )
                    st.markdown(fallback_response)
                    save_message(active_task["task_id"], "assistant", fallback_response)
                else:
                    if active_task["task_intent"] == "New Task Inquiry":
                        update_task_intent(active_task["task_id"], classification.task_title)

                    with st.spinner(f"Executing '{classification.intent}' workflow (Confidence: {classification.confidence:.2f})..."):
                        execution_result = execute_task_workflow(
                            classification.intent,
                            user_input,
                            messages
                        )

                    final_response = (
                        f"✅ **Workflow Executed:** `{classification.intent.replace('_', ' ').title()}` "
                        f"*(Confidence: {classification.confidence:.2f})*\n\n---\n\n"
                        f"{execution_result}"
                    )
                    st.markdown(final_response)
                    save_message(active_task["task_id"], "assistant", final_response)

                    if active_task["task_intent"] == "New Task Inquiry":
                        st.rerun()

            except Exception as e:
                err_message = f"❌ **System Error:** Failed to process request across available API keys and models.\n\n`{str(e)}`"
                st.markdown(err_message)
                save_message(active_task["task_id"], "assistant", err_message)