from agentqa.guards.redaction import REDACTED, find_secrets, redact, redact_text


def test_redacts_keys_and_pii() -> None:
    text = "key sk-ant-abcdefghijklmnop123 mail bob@example.com Bearer abc.def.ghi123"
    out = redact_text(text)
    assert "sk-ant" not in out and "bob@example.com" not in out and "abc.def" not in out
    assert out.count(REDACTED) == 3


def test_redacts_sensitive_keys_recursively() -> None:
    data = {"headers": {"Authorization": "Bearer x", "Accept": "json"}, "body": ["a@b.co"]}
    out = redact(data)
    assert out["headers"]["Authorization"] == REDACTED
    assert out["headers"]["Accept"] == "json"
    assert out["body"] == [REDACTED]


def test_find_secrets() -> None:
    assert find_secrets("x = 'AIza" + "A" * 35 + "'") == ["google_api_key"]
    assert find_secrets("nothing here") == []
