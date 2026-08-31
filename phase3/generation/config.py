# Default Decoding Parameters for Phase 3 Generation Experiments
#
# We fix temperature=0 to ensure deterministic generation of pull request
# descriptions while maintaining each model's default top_p value, in line
# with recommendations from some model providers (e.g., OpenAI, Alibaba Cloud).

DECODING_PARAMS = {
    "temperature": 1,
    "max_new_tokens": 1024,
    "do_sample": False,
}
