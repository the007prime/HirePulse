"""HirePulse Scope Guardrails and Input Validation Module.

Enforces strict domain boundaries to ensure HirePulse only operates within
its designated domain: AI Career Coaching, Resume Optimization, and Job Hunting.
All queries outside this scope (e.g. general trivia, creative writing, cooking,
sports, entertainment, academic homework, unrelated coding, prompt injection)
are strictly intercepted and refused.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

STANDARD_REFUSAL_MESSAGE = (
    "<!-- out_of_context -->\n"
    "I am **HirePulse**, your AI Career Coach and Resume Optimizer. 🎯\n\n"
    "I specialize strictly in career and job search development:\n"
    "- **📄 Resume & CV Optimization**: ATS keyword tailoring, formatting, impact metrics, and scoring.\n"
    "- **🔍 Job Search & Market Intelligence**: Finding matching job openings, company hiring trends, and salary benchmarks.\n"
    "- **💼 Interview Preparation**: Behavioral questions (STAR method), technical interview prep, and salary negotiation.\n"
    "- **🚀 Career Strategy & GitHub Evaluation**: Evaluating your GitHub profile, repository READMEs, portfolio, and career transitions.\n"
    "- **📁 Document Analysis**: In-depth review of your uploaded resumes, `.md` files, READMEs, or job descriptions.\n\n"
    "Feel free to upload your resume or README, share your target roles, or provide a GitHub profile link to evaluate!"
)

JAILBREAK_REFUSAL_MESSAGE = (
    "<!-- out_of_context -->\n"
    "I am **HirePulse**, an AI Career Coach and Resume Optimizer. I operate strictly within "
    "my designated career coaching scope and security guidelines. I cannot bypass my instructions, "
    "override safety boundaries, disclose internal system prompts, or adopt external personas.\n\n"
    "How can I assist you with your resume, job search, or career goals today?"
)


@dataclass
class ScopeResult:
    allowed: bool
    category: str  # "IN_SCOPE", "OUT_OF_SCOPE", "JAILBREAK_ATTEMPT"
    reason: str
    refusal_message: Optional[str] = None
    route_category: str = "IN_SCOPE"  # "IN_SCOPE" | "OUT_OF_SCOPE"


# -------------------------------------------------------------------------
# Regex Pattern Definitions
# -------------------------------------------------------------------------

# 1. Jailbreak & Prompt Injection Patterns
JAILBREAK_PATTERNS = [
    re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\s+(all\s+)?(previous|prior|above|system)\s+(instructions|prompts?|rules?|commands?|context)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(system\s*prompt|system\s*instructions?|reveal\s+(your\s+)?prompt|print\s+(your\s+)?(instructions?|prompt)|show\s+(me\s+)?your\s+instructions?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(in\s+|an?\s+)?(unrestricted|dan|jailbreak|developer\s+mode|root|god\s+mode|unfiltered)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(jailbreak|bypass\s+filters?|unfiltered\s+mode|disable\s+guardrails?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(pretend\s+you\s+have\s+no\s+(rules?|restrictions?|limits?|guardrails?))\b",
        re.IGNORECASE,
    ),
]

# 2. Conversational & Identity Whitelist Patterns (Greetings, Bot identity, Politeness)
GREETING_PATTERNS = [
    # Basic single/multi-word greetings & openers
    re.compile(
        r"\b(hi|hello|hey|heya|howdy|hola|greetings|good\s+(morning|afternoon|evening|day)|sup|yo|what'?s\s+up|wassup)\b",
        re.IGNORECASE,
    ),
    # How are you / check-in / pleasantries
    re.compile(
        r"\b(how\s+(are\s+you|are\s+you\s+doing|is\s+it\s+going|are\s+things|r\s+u)|how\s+do\s+you\s+do|hope\s+you('re|\s+are)\s+doing\s+well|nice\s+to\s+meet\s+you|pleasure\s+to\s+meet\s+you)\b",
        re.IGNORECASE,
    ),
    # Bot identity, capabilities, introduce yourself
    re.compile(
        r"\b(who\s+(are\s+you|made\s+you|created\s+you|built\s+you)|what\s+(are\s+you|can\s+you\s+do|do\s+you\s+do|is\s+your\s+name|is\s+hirepulse|is\s+this(\s+app|\s+bot)?)|tell\s+me\s+about\s+yourself|introduce\s+yourself|what\s+can\s+i\s+ask\s+you|are\s+you\s+(an?\s+)?(ai|bot|robot|assistant|human|real))\b",
        re.IGNORECASE,
    ),
    # Asking for help / assistance in general
    re.compile(
        r"\b((can|could|will)\s+you\s+help(\s+me)?(\s+with\s+something)?|how\s+can\s+you\s+help|how\s+(does\s+this|do\s+i)\s+(work|use\s+this)|i\s+need\s+(some\s+)?help|help\s+me)\b",
        re.IGNORECASE,
    ),
    # Conversational pleasantries, acknowledgments, closings
    re.compile(
        r"^\s*(ok|okay|k|cool|great|awesome|nice|got\s+it|understood|sure|sounds\s+good|perfect|alright|all\s+right|yep|yes|no|fine)[.!?\s]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(thanks?|thank\s+you(\s+so\s+much|\s+very\s+much)?|thx|ty|cheers|much\s+appreciated|appreciate\s+it)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(bye|goodbye|see\s+ya|see\s+you(\s+later)?|have\s+a\s+(good|great|nice)\s+day|take\s+care|good\s*night)\b",
        re.IGNORECASE,
    ),
]

# 3. Explicit Career & Job Domain Keywords
CAREER_PATTERNS = [
    # Resume / CV / ATS
    re.compile(
        r"\b(resume|resumes|cv|curriculum\s+vitae|ats|applicant\s+tracking|bullet\s*points?|action\s+verbs?|summary\s+section|work\s+experience)\b",
        re.IGNORECASE,
    ),
    # Job search & Career
    re.compile(
        r"\b(job|jobs|hiring|hire|vacancy|vacancies|openings?|career|careers|internship|internships|full-time|part-time|remote\s+work|hybrid\s+work|workplace)\b",
        re.IGNORECASE,
    ),
    # Compensation & Offers
    re.compile(
        r"\b(salary|salaries|compensation|pay\s*scale|remuneration|bonus|equity|stock\s+options?|job\s+offer|counteroffer|negotiat(e|ing|ion))\b",
        re.IGNORECASE,
    ),
    # Interviews & Applications
    re.compile(
        r"\b(interview|interviews|interviewing|mock\s+interview|behavioral\s+question|star\s+method|hiring\s+manager|recruiter|headhunter|cover\s+letter|application|apply|applying)\b",
        re.IGNORECASE,
    ),
    # Professional development & Role profiles & GitHub
    re.compile(
        r"\b(linkedin|portfolio|github(\.com)?|github\s+profile|github\s+repo|github\s+account|evaluate\s+(my\s+)?github|check\s+(my\s+)?github|review\s+(my\s+)?github|readme(\.md)?|upskill(ing)?|certification|certifications|transition|transitioning|career\s+change|pivot|promotion|promotion\s+path|job\s+description|jd)\b",
        re.IGNORECASE,
    ),
    # Job titles / Engineering roles / Skills
    re.compile(
        r"\b(software|engineer|engineering|developer|development|devops|sre|data\s+scientist|data\s+analyst|data\s+engineer|data\s+engineering|product\s+manager|scrum\s+master|frontend|backend|fullstack|qa|quality\s+assurance|testing|tech\s+lead|engineering\s+manager|architect|cloud\s+architect)\b",
        re.IGNORECASE,
    ),
]

# 4. Clear Out-of-Scope Topics (Prohibited domains)
OUT_OF_SCOPE_PATTERNS = [
    # Cooking / Food Recipes
    re.compile(
        r"\b(recipe|recipes|cook|cooking|bake|baking|ingredients|tablespoon|teaspoon|cuisine|dishwasher|air\s*fryer|pasta|cake|soup|sourdough)\b",
        re.IGNORECASE,
    ),
    # Sports & Athletic Games
    re.compile(
        r"\b(fifa|nba|nfl|premier\s+league|world\s+cup|cricket|basketball|baseball|soccer|football|lakers|super\s+bowl|touchdown|golden\s+boot|champions\s+league|tennis\s+match|formula\s*1|f1\s+race|game\s+score)\b",
        re.IGNORECASE,
    ),
    # Creative fiction / Poetry / Non-career entertainment
    re.compile(
        r"\b(((write|tell)(\s+me)?\s+a\s+(poem|poetry|song|lyrics|rap|rhyme|fiction|fairy\s*tale|bedtime\s+story|story|joke))|tell\s+a\s+joke|(plot|ending)\s+.*?\s+(movie|film|series)|movie\s+plot|spoilers?\s+for|actor\s+in|box\s+office|celebrity\s+gossip)\b",
        re.IGNORECASE,
    ),
    # Astronomy / Physics / Math Homework
    re.compile(
        r"\b(solve\s+for\s+x|calculate\s+the\s+derivative|integral\s+of|quadratic\s+equation|astronomy|solar\s+system|photosynthesis|mitochondria|quantum\s+physics)\b",
        re.IGNORECASE,
    ),
    # Medical & Health diagnosis
    re.compile(
        r"\b(diagnose\s+my|symptoms?\s+of|medicine\s+for|dosage\s+of|prescription\s+for|cure\s+for|remedy\s+for\s+(fever|cough|cold|infection))\b",
        re.IGNORECASE,
    ),
    # Astrology / Horoscope / Divination
    re.compile(
        r"\b(horoscope|zodiac\s+sign|astrology|tarot|palm\s+reading)\b",
        re.IGNORECASE,
    ),
    # General non-career coding / game building scripts
    re.compile(
        r"\b(snake\s+game|tic-tac-toe|flappy\s*bird|discord\s*bot|telegram\s*bot|chess\s*game|calculator\s*app|keylogger|exploit)\b",
        re.IGNORECASE,
    ),
]

# 5. Queries referencing uploaded documents
DOCUMENT_REFERENCE_PATTERNS = [
    re.compile(
        r"\b(this\s+file|uploaded\s+file|attached\s+file|the\s+document|my\s+document|the\s+pdf|uploaded\s+doc|my\s+resume|this\s+resume|the\s+attached)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(summarize|explain|review|analyze|what\s+does\s+it\s+say|what\s+is\s+in|check)\s+(this|the|my)\s+(file|doc|document|resume|upload)",
        re.IGNORECASE,
    ),
]


def is_jailbreak(text: str) -> bool:
    """Check if input contains prompt injection, persona manipulation, or jailbreak patterns."""
    for pattern in JAILBREAK_PATTERNS:
        if pattern.search(text):
            return True
    return False


def is_greeting_or_identity(text: str) -> bool:
    """Check if input is a conversational greeting, identity inquiry, or politeness."""
    cleaned = text.strip()
    for pattern in GREETING_PATTERNS:
        if pattern.search(cleaned):
            return True
    return False


def is_document_reference(text: str, attached_files: Optional[List[str]] = None) -> bool:
    """Check if query is referencing uploaded documents in this conversation."""
    if attached_files and len(attached_files) > 0:
        # If files exist, general questions about them are allowed
        for pattern in DOCUMENT_REFERENCE_PATTERNS:
            if pattern.search(text):
                return True
        # If user explicitly mentions the filename
        for f in attached_files:
            if f.lower() in text.lower():
                return True
    return False


def has_career_intent(text: str) -> bool:
    """Check if input contains explicit career, resume, or job-related keywords."""
    for pattern in CAREER_PATTERNS:
        if pattern.search(text):
            return True
    return False


def is_clear_out_of_scope(text: str) -> bool:
    """Check if input falls into an unambiguously prohibited out-of-scope domain."""
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if pattern.search(text):
            return True
    return False


def is_career_web_query(query: str) -> bool:
    """Validate whether an agent web search query is strictly career-related.
    
    Used to prevent tool misuse for non-career queries.
    """
    if not query or not query.strip():
        return False
    # Jailbreak check
    if is_jailbreak(query):
        return False
    # Clear out of scope check
    if is_clear_out_of_scope(query):
        return False
    # Must have career, hiring, salary, company, or job context
    career_web_pattern = re.compile(
        r"\b(job|jobs|hiring|layoff|layoffs|salary|salaries|career|resume|interview|compensation|tech|engineer|developer|company|workplace|remote|hybrid|openings?|internship|skills?|culture|glassdoor|levels\.fyi|ai|artificial\s+intelligence|machine\s+learning|software|github|github\.com|readme|portfolio)\b",
        re.IGNORECASE,
    )
    return bool(career_web_pattern.search(query))


def evaluate_scope(
    message: str,
    attached_files: Optional[List[str]] = None,
) -> ScopeResult:
    """Deterministic scope helper for unit tests and tool validation."""
    if not message or not message.strip():
        return ScopeResult(
            allowed=True,
            category="IN_SCOPE",
            reason="Empty message allowed",
            route_category="IN_SCOPE",
        )

    clean_msg = message.strip()

    if is_jailbreak(clean_msg):
        logger.warning(f"Guardrail intercepted jailbreak attempt: {clean_msg[:80]}")
        return ScopeResult(
            allowed=False,
            category="JAILBREAK_ATTEMPT",
            reason="Input contains prompt injection or jailbreak patterns",
            refusal_message=JAILBREAK_REFUSAL_MESSAGE,
            route_category="OUT_OF_SCOPE",
        )

    if is_document_reference(clean_msg, attached_files):
        return ScopeResult(
            allowed=True,
            category="IN_SCOPE",
            reason="User is referencing uploaded documents in this conversation",
            route_category="IN_SCOPE",
        )

    has_career = has_career_intent(clean_msg)
    is_prohibited = is_clear_out_of_scope(clean_msg)

    if is_prohibited and not has_career:
        return ScopeResult(
            allowed=False,
            category="OUT_OF_SCOPE",
            reason="Query belongs to a prohibited topic (e.g. recipes, sports, entertainment, non-career tasks)",
            refusal_message=STANDARD_REFUSAL_MESSAGE,
            route_category="OUT_OF_SCOPE",
        )

    if has_career and not is_prohibited:
        return ScopeResult(
            allowed=True,
            category="IN_SCOPE",
            reason="Query contains explicit job, career, or resume intent",
            route_category="IN_SCOPE",
        )

    if is_greeting_or_identity(clean_msg):
        return ScopeResult(
            allowed=True,
            category="IN_SCOPE",
            reason="Conversational greeting, small talk, or identity query",
            route_category="IN_SCOPE",
        )

    return ScopeResult(
        allowed=False,
        category="OUT_OF_SCOPE",
        reason="Query lacks career or job context",
        refusal_message=STANDARD_REFUSAL_MESSAGE,
        route_category="OUT_OF_SCOPE",
    )
