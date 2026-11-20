# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar, cast
from collections.abc import Callable



# ================================================================
# 1. Section: Class definition
# ================================================================
class Registry[T]:
    _entries: ClassVar[dict[str, type]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls._entries = {}

    @classmethod
    def register(cls, key: str) -> Callable[[type[T]], type[T]]:
        def decorator(target_cls: type[T]) -> type[T]:
            if key in cls._entries:
                raise ValueError(f"Key {key!r} already registered in {cls.__name__}")
            cls._entries[key] = target_cls
            return target_cls

        return decorator

    @classmethod
    def get(cls, key: str) -> T:
        return cast(T, cls.get_type(key)())

    @classmethod
    def get_type(cls, key: str) -> type[T]:
        try:
            target_cls = cls._entries[key]
        except KeyError:
            raise ValueError(
                f"Unknown key {key!r} in {cls.__name__}. "
                f"Registered: {sorted(cls._entries)}"
            ) from None
        return cast(type[T], target_cls)

    @classmethod
    def known_keys(cls) -> set[str]:
        return set(cls._entries)
