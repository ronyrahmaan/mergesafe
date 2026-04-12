"""Backdoor attack implementations for model merging."""

from mergesafe.attacks.adaptive import AdaptiveConfig, AdaptiveMode, run_adaptive_attack
from mergesafe.attacks.badnets import BadNetsAttack
from mergesafe.attacks.base import BackdoorAttack
from mergesafe.attacks.lobam import amplify_lora_weights
from mergesafe.attacks.registry import ATTACK_REGISTRY, get_attack
from mergesafe.attacks.sleeper import SleeperAttack
from mergesafe.attacks.wanet import WaNetAttack

__all__ = [
    "AdaptiveConfig",
    "AdaptiveMode",
    "BackdoorAttack",
    "BadNetsAttack",
    "WaNetAttack",
    "SleeperAttack",
    "ATTACK_REGISTRY",
    "get_attack",
    "amplify_lora_weights",
    "run_adaptive_attack",
]
