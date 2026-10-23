"""The distributed Code package must support its advertised public imports."""
from importlib import import_module


def test_code_episode_public_exports_are_importable():
    module = import_module('mandol_competition.code.software')
    exported = {name: getattr(module, name) for name in module.__all__}
    assert {'CodeAnchorExtractor', 'EngineeringEpisodeMemory',
            'EngineeringEpisodeDistiller'} <= exported.keys()
