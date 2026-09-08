"""plan_core — shared plan contract for the planner and executor agents."""
__version__ = "0.1.0"
PLAN_VERSION = "0.10.0"   # schema version this package implements

from .plan_models import QueryPlan
from .validators import (
    validate_plan, validate_schema, validate_semantics,
    ValidationResult, ValidationError,
)
from .compile_schema import compile_schema
from .biolink_vocab import BiolinkVocabulary, load_biolink_vocabulary
from .archetype_catalog import archetype_tags
