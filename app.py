import streamlit as st
from supabase import create_client, Client
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# ==========================================
# 1. Page Configuration & Client Initialization
# ==========================================
st.set_page_config(
    page_title="AI Task Assistant - Advanced Prototype",
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
    """Retrieve up to 5 Gemini API keys from Streamlit secrets with rotation support."""
    keys = []
    for i in range(1, 6):
        key_name = f"GEMINI_API_KEY_{i}"
        if key_name in st.secrets and st.secrets[key_name]:
            keys.append(st.secrets[key_name])
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
    Rotates through up to 5 API keys. Tries 'gemini-3.5-flash' first,
    falls back to 'gemini-3.8-flash' if the model fails.
    """
    api_keys = get_available_api_keys()
    if not api_keys:
        raise ValueError("No Gemini API keys found in Streamlit secrets!")

    models_to_try = ["gemini-3.5-flash", "gemini-3.8-flash"]
    last_exception = None

    for api_key in api_keys:
        try:
            client = genai.Client(api_key=api_key)
            for model_name in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    return response.text
                except Exception as model_err:
                    last_exception = model_err
                    continue
        except Exception as key_err:
            last_exception = key_err
            continue

    raise RuntimeError(
        f"All API keys and models exhausted. Last encountered error: {last_exception}"
    )

# ==========================================
# 3. Pydantic Schemas for Two-Stage AI Agent
# ==========================================
class RequestValidation(BaseModel):
    has_contradiction: bool = Field(
        description="True if the user request directly contradicts or violates the task's preferences_list, otherwise False."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 regarding whether AI can successfully fulfill the request."
    )
    reasoning: str = Field(
        description="Explanation of the evaluation, detailing any contradictions or reasons why confidence is low."
    )
    task_title_suggestion: str = Field(
        description="A concise 3-6 word summary of the task intent."
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
    default_prefs = "Provide professional, structured, and accurate results matching user specifications."
    task_res = (
        supabase.table("tasks")
        .insert({
            "user_id": user_id,
            "task_intent": "New Task Inquiry",
            "preferences_list": default_prefs
        })
        .execute()
    )
    new_task = task_res.data[0]
    task_id = new_task["task_id"]

    initial_greeting = (
        "Hello! I am your AI Task Assistant. Set your task preferences in the settings panel above, "
        "and type your request below. I will validate your request against your preferences before execution!"
    )
    save_message(task_id, "assistant", initial_greeting)
    return new_task

def update_task_details(task_id: str, task_intent: str, preferences_list: str) -> None:
    """Update task title and preferences independently of chat requests."""
    supabase.table("tasks").update({
        "task_intent": task_intent,
        "preferences_list": preferences_list
    }).eq("task_id", task_id).execute()

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
# 5. Two-Stage AI Agent Workflow
# ==========================================
def node_1_validate_request(user_message: str, preferences_list: str, chat_history: List[Dict[str, Any]]) -> RequestValidation:
    """
    Stage 1: Evaluates if user request conflicts with preferences_list
    and determines confidence score.
    """
    history_context = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history[-6:]])

    system_instruction = (
        "You are a strict AI Guardrail and Intent Router. "
        "Your job is to evaluate the user's latest request against the defined 'Task Preferences List'. "
        "1. Check if the user request contradicts the preferences list. "
        "2. Evaluate if the request is clear and within AI processing capabilities (assign a confidence score 0.0 to 1.0). "
        "If confidence is below 0.7 or if there is a contradiction, set confidence < 0.7 and explain in reasoning."
    )

    prompt = (
        f"--- TASK PREFERENCES LIST ---\n{preferences_list}\n\n"
        f"--- RECENT CONVERSATION CONTEXT ---\n{history_context}\n\n"
        f"--- LATEST USER REQUEST ---\n{user_message}"
    )

    response_text = generate_with_key_rotation(
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=RequestValidation,
            temperature=0.1,
        )
    )
    return RequestValidation.model_validate_json(response_text)

def node_2_execute_task(user_message: str, preferences_list: str, chat_history: List[Dict[str, Any]]) -> str:
    """
    Stage 2: Executes the task while strictly adhering to preferences_list.
    """
    history_context = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history[-6:]])

    system_instruction = (
        f"You are an expert AI task execution agent. "
        f"You MUST strictly adhere to the following task preferences and constraints at all times:\n"
        f"=== TASK PREFERENCES ===\n{preferences_list}\n========================"
    )

    prompt = f"Conversation Context:\n{history_context}\n\nUser Request to Execute:\n{user_message}"

    return generate_with_key_rotation(
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
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
    st.caption("Advanced Two-Stage AI Agent")

    user_id = st.text_input("Active User ID", value="demo_user_001").strip()

    if not user_id:
        st.warning("Please enter a User ID to load tasks.")
        st.stop()

    st.divider()

    if st.button("➕ Create New Task", use_container_width=True, type="primary"):
        created_task = create_new_task(user_id)
        st.session_state.active_task_id = created_task["task_id"]
        st.rerun()

    st.subheader("Your Tasks")
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
            if st.button(button_label, key=f"select_{t_id}", use_container_width=True, help=f"ID: {t_id}"):
                st.session_state.active_task_id = t_id
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{t_id}", help="Delete task"):
                delete_task(t_id)
                if st.session_state.active_task_id == t_id:
                    st.session_state.active_task_id = None
                st.rerun()

# --- Main Area: Active Task Workspace ---
active_task = next((t for t in user_tasks if t["task_id"] == st.session_state.active_task_id), None)

if active_task:
    st.header(f"Task: {active_task['task_intent']}")
    st.caption(f"**User ID:** `{user_id}` | **Task ID (Read-only):** `{active_task['task_id']}`")

    # --- Task Settings Panel (Edit Title & Preferences Separately) ---
    with st.expander("⚙️ Task Settings & Preferences Configuration", expanded=False):
        with st.form(key=f"settings_form_{active_task['task_id']}"):
            new_title = st.text_input("Task Title", value=active_task["task_intent"])
            new_preferences = st.text_area(
                "Preferences List (Constraints, formatting rules, target language, tone, etc.)",
                value=active_task.get("preferences_list", "")
            )
            submit_settings = st.form_submit_button("Save Task Settings", type="primary")
            
            if submit_settings:
                update_task_details(active_task["task_id"], new_title, new_preferences)
                st.success("Task settings updated successfully!")
                st.rerun()

    st.divider()

    # --- Conversation History Display ---
    messages = get_task_messages(active_task["task_id"])
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat Input (Triggers Two-Stage Agent Workflow) ---
    if user_input := st.chat_input("Enter your request for this task..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        save_message(active_task["task_id"], "user", user_input)

        with st.chat_message("assistant"):
            try:
                # Stage 1: Validation Node
                with st.spinner("Stage 1: Validating request against preferences & checking confidence..."):
                    validation = node_1_validate_request(
                        user_input,
                        active_task.get("preferences_list", ""),
                        messages
                    )

                # Check Confidence Threshold (< 0.7) or Contradiction
                if validation.has_contradiction or validation.confidence < 0.7:
                    fallback_response = (
                        f"⚠️ **Cannot Process Request (Human Assistance Required)**\n\n"
                        f"- **Confidence Score:** `{validation.confidence:.2f}` (Threshold: `0.70`)\n"
                        f"- **Has Contradiction:** `{str(validation.has_contradiction)}`\n\n"
                        f"**Reasoning:** {validation.reasoning}\n\n"
                        f"Please adjust your task preferences or provide clearer instructions so human assistance or a revised request can proceed."
                    )
                    st.markdown(fallback_response)
                    save_message(active_task["task_id"], "assistant", fallback_response)
                else:
                    # Auto-update task title if it was still default
                    if active_task["task_intent"] == "New Task Inquiry":
                        update_task_details(active_task["task_id"], validation.task_title_suggestion, active_task.get("preferences_list", ""))

                    # Stage 2: Execution Node (Strictly adheres to preferences)
                    with st.spinner(f"Stage 2: Executing workflow (Confidence: {validation.confidence:.2f})..."):
                        execution_result = node_2_execute_task(
                            user_input,
                            active_task.get("preferences_list", ""),
                            messages
                        )

                    final_response = (
                        f"✅ **Validated & Executed** *(Confidence: {validation.confidence:.2f})*\n\n---\n\n"
                        f"{execution_result}"
                    )
                    st.markdown(final_response)
                    save_message(active_task["task_id"], "assistant", final_response)

                    if active_task["task_intent"] == "New Task Inquiry":
                        st.rerun()

            except Exception as e:
                err_message = f"❌ **System Error:** Failed during two-stage evaluation.\n\n`{str(e)}`"
                st.markdown(err_message)
                save_message(active_task["task_id"], "assistant", err_message)