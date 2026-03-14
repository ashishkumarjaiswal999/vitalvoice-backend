"""
crew.py — Optimized 2-agent VitalVoice pipeline.
Agent 1: Intent Filter + RAG Retrieval
Agent 2: Medical Analysis + Response Generation
50% fewer tokens than 4-agent version, still genuinely multi-agent.
"""

from crewai import Agent, Task, Crew, LLM, Process
from rag import retrieve_medical_context
import os

BLOCKED_RESPONSE = (
    "VitalVoice is designed to answer healthcare-related questions only. "
    "Please ask me about symptoms, medicines, health conditions, "
    "mental health, nutrition, or any other health topic."
)

# ── Fast local keyword block — zero API calls ──────────────────────────────
NON_HEALTH_KEYWORDS = [
    "python", "java", "code", "programming", "function", "loop", "array",
    "algorithm", "sql", "html", "css", "javascript", "politics", "election",
    "finance", "stock", "movie", "song", "cricket", "football", "recipe",
    "travel", "hotel", "weather", "print", "variable", "class", "object"
]

def is_blocked_locally(text: str) -> bool:
    """Instant local check — no API call needed for obvious non-health queries."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in NON_HEALTH_KEYWORDS)

def get_llm():
    return LLM(
        model="gemini/gemini-3-flash-preview",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.4,
        max_tokens=250      # Keep each agent response short
    )

class VitalVoiceCrew:

    def __init__(self):
        self.llm = get_llm()

    def run(self, user_input: str, module: str = "general") -> dict:
        """
        2-agent pipeline:
        Agent 1 — Intent Filter + RAG context retrieval
        Agent 2 — Medical analysis + safe response generation
        """

        # ── Step 1: Local block (0 API calls) ─────────────────────────
        if is_blocked_locally(user_input):
            return {
                "response": BLOCKED_RESPONSE,
                "blocked": True,
                "module": module
            }

        # ── Step 2: Retrieve RAG context from ChromaDB ────────────────
        medical_context = retrieve_medical_context(user_input, n_results=2)

        # ── Agent 1: Intent Filter + RAG Retriever ────────────────────
        retriever_agent = Agent(
            role="Medical Intent Filter and Knowledge Retriever",
            goal="Verify the query is health-related and summarize relevant medical context.",
            backstory="""You are the first line of defense for VitalVoice.
            You have two jobs:
            1. Confirm the query is health-related. If not, respond BLOCKED.
            2. If health-related, extract and summarize the most relevant
               facts from the provided medical knowledge base.
            Be concise — your output feeds directly into the next agent.""",
            llm=self.llm,
            verbose=False,
            allow_delegation=False
        )

        # ── Agent 2: Medical Analyst + Response Writer ─────────────────
        responder_agent = Agent(
            role="Medical Analyst and Patient Communicator",
            goal="Analyze health queries and write clear, safe, actionable responses.",
            backstory="""You are an experienced medical professional and patient communicator.
            You receive a verified health query and relevant medical context.
            You provide accurate, compassionate, concise health guidance.
            You always flag emergencies and recommend professional consultation.
            You write in simple language suitable for rural and low-literacy users.""",
            llm=self.llm,
            verbose=False,
            allow_delegation=False
        )

        # ── Task 1: Intent + RAG ───────────────────────────────────────
        retriever_task = Task(
            description=f"""Query: "{user_input}"

Retrieved medical knowledge:
{medical_context}

Step 1: Is this health-related? If NO → respond only: BLOCKED
Step 2: If YES → summarize the 2-3 most relevant medical facts
        from the knowledge above that apply to this query.
        Also note if any emergency indicators are present.
Keep your response under 100 words.""",
            agent=retriever_agent,
            expected_output="BLOCKED or a concise summary of relevant medical facts under 100 words"
        )

        # ── Task 2: Medical Response ───────────────────────────────────
        responder_task = Task(
            description=f"""Patient query: "{user_input}"
Module: {module}

Using the medical context from the previous agent, write a response that:
1. Acknowledges the patient's concern (1 sentence)
2. Explains what it could mean simply (1-2 sentences)
3. Gives 2-3 practical steps as bullet points
4. If emergency detected: start with URGENT: and advise immediate care
5. Ends with: "Note: This is AI health guidance. Please consult a doctor."

Maximum 150 words. Simple language.""",
            agent=responder_agent,
            expected_output="Concise compassionate health response under 150 words",
            context=[retriever_task]
        )

        # ── Run 2-agent Crew ───────────────────────────────────────────
        try:
            crew = Crew(
                agents=[retriever_agent, responder_agent],
                tasks=[retriever_task, responder_task],
                process=Process.sequential,
                verbose=False
            )

            result = crew.kickoff()
            retriever_output = str(retriever_task.output).upper() if retriever_task.output else ""

            # Check if blocked by Agent 1
            if "BLOCKED" in retriever_output:
                return {
                    "response": BLOCKED_RESPONSE,
                    "blocked": True,
                    "module": module
                }

            return {
                "response": str(result),
                "blocked": False,
                "module": module
            }

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                return {
                    "response": "API quota reached. Please wait a minute and try again.",
                    "blocked": False,
                    "module": module
                }
            return {
                "response": f"Something went wrong. Please try again.\n({error_msg[:80]})",
                "blocked": False,
                "module": module
            }
