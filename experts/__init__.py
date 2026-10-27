"""
Importing this package registers the builtin experts in EXPERT_REGISTRY
(each module calls register() on import). To add a new builtin expert just
import its module here.
"""
# builtin experts, importing each one registers it
from experts import price_macro   # noqa: F401  (registers "pm")
from experts import sentiment     # noqa: F401  (registers "sn")
from experts import fundamental   # noqa: F401  (registers "fn")
