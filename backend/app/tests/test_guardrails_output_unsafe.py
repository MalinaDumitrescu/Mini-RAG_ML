from backend.app.services.guardrails import check_output

def test_guardrails_blocks_unsafe_output():
    r = check_output("Here is porn content ...")
    assert r["ok"] is False
    assert r["reason"] in ("unsafe_output", "unsafe_output_prompt_leak_or_injection")
