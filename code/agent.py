"""
Support triage agent using LangChain + Groq.

For each ticket it:
1. Retrieves relevant support corpus documents
2. Classifies the request type, product area, urgency, and risk
3. Decides whether to reply or escalate
4. Generates a safe, grounded response
"""

import json
import os
import re

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from retriever import retrieve_relevant_docs


GROQ_MODEL = "llama-3.3-70b-versatile"

TRIAGE_SYSTEM_PROMPT = """\
You are a support triage agent for three product ecosystems:
- **HackerRank**: coding assessments, tests, interviews, certifications, community platform
- **Claude** (by Anthropic): AI assistant, API, subscriptions, team/enterprise plans
- **Visa**: credit/debit cards, travel support, fraud, payments

Your task: analyze a support ticket and produce a structured JSON triage decision.

═══ STRICT RULES ═══

1. **Ground every claim in the CORPUS DOCUMENTS below.** Never invent policies, URLs, phone numbers, or steps that are not explicitly present in the provided documents. If the corpus lacks the answer, escalate.

2. **Escalation triggers** — set status to "escalated" for ANY of these:
   - Billing disputes, refund demands with threatening language, or payment fraud
   - Account security breaches, identity theft, unauthorized access
   - Legal threats or regulatory complaints
   - Requests to change test scores, override hiring decisions, or alter assessment results
   - Requests requiring admin/backend access that cannot be self-served
   - Abusive, threatening, or harassing language toward support staff
   - The corpus provides NO relevant information to answer the question
   - Vague "site is down" or outage reports with no clear product identified
   - Requests involving sensitive personal data (SSN, full card numbers, etc.)
   When escalating, set response to a brief message like: "This issue requires human review. Escalating to our support team for further assistance."

3. **Out-of-scope requests** — if the issue is completely unrelated to HackerRank, Claude, or Visa:
   - Set status to "replied", request_type to "invalid"
   - Respond politely: "I'm sorry, this falls outside the scope of our support capabilities."

4. **Simple/pleasantry messages** (e.g., "thank you", "hello"):
   - Set status to "replied", request_type to "invalid"
   - Respond warmly but briefly.

5. **Product area** — use a SHORT label for the relevant domain area. Examples:
   - HackerRank: "screen", "interviews", "community", "certifications", "library", "settings", "integrations", "skillup", "engage", "chakra"
   - Claude: "account_management", "billing", "api", "privacy", "troubleshooting", "team_plans", "enterprise", "claude_desktop", "claude_code", "connectors"
   - Visa: "general_support", "travel_support", "fraud_protection", "consumer_support"
   Pick the MOST relevant short label. Do not use long paths.

6. **Request type classification**:
   - "product_issue": general usage questions, how-to, configuration, account management
   - "feature_request": user asking for new features or capabilities
   - "bug": user reports something broken, errors, crashes, unexpected behavior
   - "invalid": off-topic, spam, pleasantries, or completely unrelated requests

7. **Response style**: Be concise, professional, and actionable. Start with a greeting only for longer responses. Provide step-by-step instructions when the corpus has them.

═══ CORPUS DOCUMENTS ═══
{context}

═══ TICKET ═══
Issue: {issue}
Subject: {subject}
Company: {company}

═══ OUTPUT FORMAT ═══
Respond with ONLY a valid JSON object (no markdown fences, no extra text):
{{
  "status": "replied" or "escalated",
  "product_area": "<short category label>",
  "response": "<user-facing answer or escalation message>",
  "justification": "<1-2 sentence explanation of your decision>",
  "request_type": "product_issue" or "feature_request" or "bug" or "invalid"
}}
"""

_llm_cache = None


def get_llm() -> ChatGroq:
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    _llm_cache = ChatGroq(
        model=GROQ_MODEL,
        api_key=api_key,
        temperature=0.0,
        max_tokens=2048,
    )
    return _llm_cache


def format_context(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        parts.append(
            f"[Doc {i}] Domain: {meta.get('domain', 'Unknown')} | "
            f"Category: {meta.get('category', 'Unknown')}\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def parse_agent_output(raw: str) -> dict:
    """Parse the JSON output from the LLM, handling common formatting issues."""
    raw = raw.strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
        raw = raw.strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "status": "escalated",
            "product_area": "unknown",
            "response": "This issue requires human review. Escalating to our support team.",
            "justification": "Agent output parsing failed; escalating for safety.",
            "request_type": "product_issue",
        }

    result["status"] = result.get("status", "escalated").lower().strip()
    if result["status"] not in ("replied", "escalated"):
        result["status"] = "escalated"

    result["request_type"] = result.get("request_type", "product_issue").lower().strip()
    if result["request_type"] not in ("product_issue", "feature_request", "bug", "invalid"):
        result["request_type"] = "product_issue"

    for key in ("product_area", "response", "justification"):
        result[key] = result.get(key, "").strip()

    return result


def triage_ticket(
    store: FAISS,
    issue: str,
    subject: str,
    company: str,
) -> dict:
    """Process a single support ticket through the triage agent."""
    llm = get_llm()

    query = f"{subject} {issue}".strip() if subject else issue

    domain = company if company and company != "None" else None
    docs = retrieve_relevant_docs(store, query, domain=domain, k=6)

    context = format_context(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", TRIAGE_SYSTEM_PROMPT),
    ])

    chain = prompt | llm | StrOutputParser()

    raw_output = chain.invoke({
        "context": context,
        "issue": issue,
        "subject": subject or "(no subject)",
        "company": company or "None",
    })

    return parse_agent_output(raw_output)
