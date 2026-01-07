from backend.app.rag.llm import GeneratorLLM, LLMConfig

def test_generator_initializes():
    llm = GeneratorLLM(LLMConfig(device="cpu"))  # force CPU for CI/laptops
    out = llm.generate_chat("You are a helpful assistant.", "Say OK.")
    assert isinstance(out, str)
    assert len(out) > 0
