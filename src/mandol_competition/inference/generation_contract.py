"""Explicit, fingerprintable generative identity; legacy defaults remain valid."""
import re


def validate_generation_binding(model, provider, request_model):
    for value in (model, provider, request_model):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,160}", value):
            raise ValueError("Explicit bounded generation model/provider identity required")


def verified_response_identity(record):
    actual = record.get("response_model", "")
    contract = record.get("model_contract")
    if contract is not None:
        # The transport attests this against its configured identity before emitting a receipt.
        return (isinstance(contract, dict) and contract.get("schema") == "generation-binding-v1"
                and isinstance(contract.get("accepted_response_models"), list)
                and actual in contract["accepted_response_models"]
                and bool(contract.get("logical_model")) and bool(contract.get("request_model")))
    return isinstance(actual, str) and bool(re.fullmatch(r"gpt-4o-mini(?:-\d{4}-\d{2}-\d{2})?", actual))
