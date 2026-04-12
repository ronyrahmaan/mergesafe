"""Attack registry — maps attack type names to classes."""

from typing import Any

from mergesafe.attacks.badnets import BadNetsAttack
from mergesafe.attacks.base import AttackConfig, BackdoorAttack
from mergesafe.attacks.sleeper import SleeperAttack
from mergesafe.attacks.wanet import WaNetAttack

ATTACK_REGISTRY: dict[str, type[BackdoorAttack]] = {
    "badnets": BadNetsAttack,
    "wanet": WaNetAttack,
    "sleeper": SleeperAttack,
}


def get_attack(attack_type: str, **config_overrides: Any) -> BackdoorAttack:
    """Create an attack instance by name.

    Args:
        attack_type: One of 'badnets', 'wanet', 'sleeper'.
        **config_overrides: Override AttackConfig defaults.

    Returns:
        Configured BackdoorAttack instance.
    """
    if attack_type not in ATTACK_REGISTRY:
        msg = f"Unknown attack type '{attack_type}'. Available: {list(ATTACK_REGISTRY.keys())}"
        raise ValueError(msg)

    config = AttackConfig(attack_type=attack_type, **config_overrides)
    return ATTACK_REGISTRY[attack_type](config)
