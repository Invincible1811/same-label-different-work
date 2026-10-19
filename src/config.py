"""Shared paths and study constants. Every analysis script imports from here,
so a decision changed in one place changes everywhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
FIGURE_DATA = ROOT / "results" / "figure_data"  # exact values plotted in each figure

SEED = 42

# D12: end of observation for open (censored) issues = latest api_updated_at in
# the dataset, i.e. when the GitHub API snapshot was taken.
SNAPSHOT_DATE = "2026-10-09T17:48:25Z"

# D7: minimum group sizes for inferential comparisons.
MIN_CLOSED_RQ2 = 30
MIN_DOCS_RQ3 = 30

# D8: first public availability of each agent (earliest of preview/announcement/
# first public repo), used only for the pre-release sensitivity flag.
# (date, source, confidence)
RELEASE_DATES = {
    "Aider": ("2023-05-09", "github.com/Aider-AI/aider repository created_at (GitHub API)", "high"),
    "Sweep": ("2023-06-14", "github.com/sweepai/sweep repository created_at (GitHub API)", "high"),
    "PR_Agent": ("2023-07-05", "github.com/qodo-ai/pr-agent repository created_at (GitHub API)", "high"),
    "CodeRabbit": ("2023-07-10", "Product Hunt launch listing; company founded 2023 (sacra.com)", "medium"),
    "Amazon_Q": ("2023-11-28", "AWS News Blog: Amazon Q preview at re:Invent 2023", "high"),
    "Devin": ("2024-03-12", "cognition.ai/introducing-devin", "high"),
    "SWE_agent": ("2024-04-02", "github.com/SWE-agent/SWE-agent repository created_at (GitHub API)", "high"),
    "Gemini_Code_Assist": (
        "2024-04-09",
        "Announced at Google Cloud Next '24 (thenewstack.io, virtualizationreview.com)",
        "high",
    ),
    "Jules": ("2024-12-11", "Google Gemini 2.0 announcement (Labs preview); not re-verified", "medium"),
    "Claude_Code": ("2025-02-24", "Anthropic research preview with Claude 3.7 Sonnet", "high"),
    "Copilot": ("2025-05-19", "GitHub Copilot coding agent, Microsoft Build 2025; not re-verified", "low"),
    "Kiro": ("2025-07-14", "GeekWire launch report, 14 Jul 2025", "high"),
    "Cursor": ("2023-03-14", "Cursor editor public release; not re-verified", "low"),
}

# D15: attribution signals, following the CodAGE README (Ghaleb, 2026), which is
# the upstream source of this dataset. Vendor-controlled [bot] logins are the
# high-precision signal; github-actions[bot] re-posts agent output; repositories
# named after the agent hold issues *about* the agent (self-referential).
OWN_BOT_LOGINS = {
    "CodeRabbit": {"coderabbitai[bot]", "coderabbit[bot]"},
    "Devin": {"devin-ai-integration[bot]"},
    "Sweep": {"sweep-ai[bot]"},
    "Gemini_Code_Assist": {"gemini-code-assist[bot]"},
    "Amazon_Q": {"amazon-q-developer[bot]"},
    "Jules": {"jules-google[bot]"},
    "PR_Agent": {"codiumai-pr-agent[bot]", "pr-agent[bot]"},
    "Copilot": {"github-copilot[bot]"},
    # Not in the CodAGE login list; claude[bot] is the Claude GitHub App.
    "Claude_Code": {"claude[bot]"},
}
AUTOMATION_LOGINS = {"github-actions[bot]"}
SELF_REF_REPO_PATTERNS = {
    "SWE_agent": r"swe-?agent|swe-?bench|swe-?smith",
    "Gemini_Code_Assist": r"gemini",
    "Amazon_Q": r"amazon-q|q-developer",
    "Aider": r"aider",
    "Claude_Code": r"claude-?code|claude_code",
    "Sweep": r"sweep",
    "Kiro": r"kiro",
    "Jules": r"jules",
    "Devin": r"devin",
    "CodeRabbit": r"coderabbit",
    "PR_Agent": r"pr-agent|qodo|codium",
    "Copilot": r"copilot",
    "Cursor": r"cursor",
}

# RQ3 text preprocessing (LDA on English issues). Every rule is listed here so it
# can be cited and replicated.
RQ3_TEMPLATE_MIN_DOCS = 10  # a masked line repeated verbatim in >= 10 issues is template text
RQ3_MIN_TOKENS = 10  # documents shorter than this after cleaning are not modelled
RQ3_SIGNATURE_LINES = [  # agent signatures / triggers (CodAGE attribution signals) -> removed
    r"co-?authored-?by:",
    r"generated (with|by) .*(claude|devin|kiro|jules|cursor|codex|pr-agent|copilot)",
    r"reviewed by coderabbit",
    r"summary by coderabbit",
    r"coderabbit",
    r"/q dev",
    r"/sweep\b",
    r"devin run",
    r"link to devin",
    r"requested by:",
    r"built with jules",
]
RQ3_AGENT_PHRASES = [  # multi-word product names, removed at text level before tokenising
    r"amazon\s+q(\s+developer)?",
    r"\bq\s+developer",
    r"gemini\s+code\s+assist",
    r"\bcode\s+assist\b",
    r"claude\s+code",
    r"github\s+copilot",
    r"copilot\s+(coding\s+)?agent",
    r"pr[\s-]?agent",
    r"swe[\s-]?agent",
    r"sweep(\s*ai)?\b",
    r"devin(\s+ai)?",
    r"google\s+jules",
    r"kiro(\.dev)?",
    r"codiumai|qodo",
]
RQ3_AGENT_NAMES = {  # product / vendor names of the 13 agents (attribution signals) -> removed tokens
    "claude",
    "anthropic",
    "coderabbit",
    "coderabbitai",
    "rabbit",
    "sweep",
    "sweepai",
    "devin",
    "cognition",
    "sweagent",
    "swe",
    "gemini",
    "codeassist",
    "amazon",
    "codewhisperer",
    "aider",
    "kiro",
    "jules",
    "codium",
    "codiumai",
    "qodo",
    "pragent",
    "copilot",
    "cursor",
    "codex",
}
RQ3_AGENT_SUBSTRINGS = (  # unambiguous product strings: drop any token *containing* them
    "claude",
    "anthropic",
    "coderabbit",
    "gemini",
    "amazonq",
    "codewhisperer",
    "codium",
    "qodo",
    "devin",
    "sweagent",
    "sweepai",
    "copilot",
    "codex",
    "codeassist",
    "code_assist",
)  # (ambiguous names such as cursor, sweep, aider, kiro, jules: exact match only)
RQ3_EXTRA_STOPWORDS = {  # generic issue-tracker words that carry no topic
    "issue",
    "issues",
    "github",
    "com",
    "http",
    "https",
    "www",
    "html",
    "png",
    "jpg",
    "please",
    "thanks",
    "thank",
    "hello",
    "hi",
    "let",
    "know",
    "like",
    "just",
    "want",
    "need",
    "currently",
    "would",
    "could",
    "should",
    "also",
    "use",
    "using",
    "used",
    "make",
    "way",
    "get",
    "got",
    "etc",
    "e.g",
    "i.e",
    "via",
    "todo",
    "tbd",
    "n/a",
    "none",
    "yes",
    "true",
    "false",
    "null",
    "response",
    "details",
    "line",
    "lines",
}
RQ3_K_RANGE = range(5, 21)
# Topic labels: assigned by reading the top words and representative issues of each topic, then
# validated by the author on 5 random issues per fine topic (docs/validation, tab V2_topic_labels).
# {level: {topic number: label}}. The labels are kept as they were judged in the validation.
RQ3_TOPIC_LABELS = {
    "headline": {1: "Code maintenance and quality work", 2: "AI/LLM tooling and research"},
    "fine": {
        1: "Testing, review and quality follow-ups",
        3: "LLM, agent and model features",
        4: "Code-level fixes and refactoring",
        5: "Build, CI and repository workflow",
    },
}
# Topics we interpret. Rule (fine view): reproduced across the two ensembles (>= 7 of 10 top words)
# AND label confirmed by the validation (fewer than 2 of 5 sampled issues judged "does not fit",
# the threshold set before the validation). Fine topic 2 is not reproduced; topics 3 and 5 are only
# partly reproduced and their labels failed the validation (2/5 and 1/5 fit). The headline topics are
# both reproduced; they were not part of the label validation.
RQ3_INTERPRETED = {"headline": [1, 2], "fine": [1, 4]}
RQ3_PASSES, RQ3_ITERATIONS = 10, 200

# D9: label -> category. Rules are applied in order; first match wins.
# "agent_workflow" = labels that drive or mark an AI agent / automation,
# not maintainer triage (e.g. Sweep is triggered by the `sweep` label).
LABEL_RULES = [
    (
        "agent_workflow",
        r"sweep|devin|amazon[- ]?q|jules|copilot|coderabbit|gemini|claude|aider|kiro|codex|"
        r"swe[- ]?agent|pr[- ]?agent|cursor|agentic|^automated$|^automation$|bot-triaged|repo-assist",
    ),
    ("bug", r"bug|defect|regression|crash|ci-failure|broken|^fix$"),
    ("feature", r"enhancement|feature|feat\b|^feat$|improvement|request"),
    ("documentation", r"doc(?!ker)"),
]
OTHER_LABEL = "other"
