from cellflow.networks._set_encoders import (
    ConditionEncoder,
)
from cellflow.networks._utils import (
    MLPBlock,
    SeedAttentionPooling,
    SelfAttention,
    SelfAttentionBlock,
    TokenAttentionPooling,
)
from cellflow.networks._velocity_field import ConditionalVelocityField, GENOTConditionalVelocityField

try:
    from cellflow.networks._prophet_adapter import ProphetEncoder
    _PROPHET_AVAILABLE = True
except ImportError:
    _PROPHET_AVAILABLE = False

__all__ = [
    "ConditionalVelocityField",
    "GENOTConditionalVelocityField",
    "ConditionEncoder",
    "MLPBlock",
    "SelfAttention",
    "SeedAttentionPooling",
    "TokenAttentionPooling",
    "SelfAttentionBlock",
]

if _PROPHET_AVAILABLE:
    __all__ += ["ProphetEncoder"]
