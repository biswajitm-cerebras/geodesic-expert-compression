## EvalPlus
DEFAULT_GT_TIME_LIMIT_FACTOR = 4.0
DEFAULT_MIN_TIME_LIMIT = 4.0

## EvalPerf

### General
PERF_PROFILE_ROUNDS = 1
PERF_RAM_GB_PER_PROC = 12

### Evaluation Phase
PERF_EVAL_TIMEOUT_SECOND = 45
SYNTAX_CHECK_TIMEOUT_SECONDS = 5

### Curation Phase
PERF_CURATE_TIMEOUT_SECOND = 20
PREF_CURATE_MIN_INSTRUCTION = 10000

# Max new tokens default for run_codegen fn
# MAX_NEW_TOKENS = 2048 # Set to None to use the max model length
# MAX_NEW_TOKENS = 131072 # Set to None to use the max model length
MAX_NEW_TOKENS = 16384 # Set to None to use the max model length
