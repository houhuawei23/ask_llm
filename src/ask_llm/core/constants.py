"""Constants and configuration values used across the application.

This module centralizes magic numbers and hardcoded strings to improve
maintainability and make configuration more explicit.
"""

from enum import Enum


class TaskKind(str, Enum):
    """Types of batch tasks for token estimation."""

    PAPER_EXPLAIN = "paper_explain"
    TRANSLATION = "translation"
    BATCH = "batch"
    FORMAT = "format"


# Output token estimation multipliers by task type
# These are heuristic estimates based on typical task characteristics
OUTPUT_TOKEN_MULTIPLIERS = {
    TaskKind.PAPER_EXPLAIN: 2.0,  # Explanations tend to be longer than input
    TaskKind.TRANSLATION: 1.1,  # Translations typically similar length or slightly longer
    TaskKind.BATCH: 1.1,  # Generic batch tasks
    TaskKind.FORMAT: 1.0,  # Formatting typically preserves length
}

# Default fallback model when no model is specified
DEFAULT_FALLBACK_MODEL = "deepseek-reasoner"

# Minimum token estimate for tasks with zero or unknown input
DEFAULT_MIN_OUTPUT_TOKENS = 100

# Progress bar update throttle (seconds) to prevent UI flickering
PROGRESS_UPDATE_INTERVAL = 0.1

# Token counting cache size (LRU cache for performance)
TOKEN_COUNT_CACHE_SIZE = 1024

# Safety margin applied to token budgets for providers whose real BPE differs
# from the cl100k_base approximation (DeepSeek, Qwen). cl100k_base undercounts
# CJK text, so a chunk that "fits" by cl100k count can overflow the provider's
# real context window. Sizing chunks to 85% of the budget leaves headroom for
# the undercount. See ARCHITECTURE_REVIEW.md bug B2.
APPROX_TOKEN_SAFETY_FACTOR = 0.85
