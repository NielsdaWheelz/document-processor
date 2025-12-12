"""
Validation helpers for the document-processor pipeline.

Provides utilities for validating JSON data against pydantic models,
including support for list types.
"""

import json
from typing import Any, TypeVar, get_args, get_origin

from pydantic import BaseModel, TypeAdapter, ValidationError

T = TypeVar("T", bound=BaseModel)


class ValidationException(Exception):
    """Raised when validation fails."""

    def __init__(self, message: str, errors: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or []


class JSONParseException(Exception):
    """Raised when JSON parsing fails."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def load_json_bytes(b: bytes) -> Any:
    """
    Parse JSON from bytes.

    Args:
        b: Raw bytes containing JSON data

    Returns:
        Parsed JSON data

    Raises:
        JSONParseException: If JSON parsing fails
    """
    try:
        return json.loads(b.decode("utf-8"))
    except UnicodeDecodeError as e:
        raise JSONParseException(f"Invalid UTF-8 encoding: {e}")
    except json.JSONDecodeError as e:
        raise JSONParseException(f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}")


def validate_data(model_or_type: type[T] | Any, data: Any) -> T | list[Any]:
    """
    Validate data against a pydantic model or type.

    Supports both single models and list types like list[DocIndexItem].

    Args:
        model_or_type: A pydantic BaseModel class or a type hint (e.g., list[Model])
        data: The data to validate

    Returns:
        Validated and parsed object(s)

    Raises:
        ValidationException: If validation fails
    """
    try:
        # Check if this is a generic type (like list[Something])
        origin = get_origin(model_or_type)
        if origin is list:
            # Get the inner type
            args = get_args(model_or_type)
            if args:
                inner_type = args[0]
                adapter = TypeAdapter(list[inner_type])
                return adapter.validate_python(data)
            else:
                # Plain list without type args
                return list(data)

        # Check if it's a pydantic model
        if isinstance(model_or_type, type) and issubclass(model_or_type, BaseModel):
            return model_or_type.model_validate(data)

        # Fallback to TypeAdapter for other types
        adapter = TypeAdapter(model_or_type)
        return adapter.validate_python(data)

    except ValidationError as e:
        errors = e.errors()
        error_messages = []
        for err in errors:
            loc = ".".join(str(x) for x in err["loc"])
            msg = err["msg"]
            error_messages.append(f"{loc}: {msg}")
        raise ValidationException(
            f"Validation failed: {'; '.join(error_messages)}",
            errors=[dict(e) for e in errors],
        )
    except TypeError as e:
        raise ValidationException(f"Type error during validation: {e}")
