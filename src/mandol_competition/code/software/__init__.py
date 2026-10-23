"""Lazy Code exports; importing episode policy cannot load a model."""
from importlib import import_module
_EXPORTS = {
    'CodeAnchor': ('.anchors', 'CodeAnchor'),
    'CodeAnchorExtractor': ('.anchors', 'CodeAnchorExtractor'),
    'EngineeringEpisodeDistiller': ('.episode_distillation', 'EngineeringEpisodeDistiller'),
    'EpisodeDistillationResult': ('.episode_distillation', 'EpisodeDistillationResult'),
    'EngineeringEpisodeMemory': ('.episodes', 'EngineeringEpisodeMemory'),
    'EngineeringEpisodeBuilder': ('.episodes', 'EngineeringEpisodeBuilder'),
    'EngineeringAttempt': ('.episodes', 'EngineeringAttempt'),
    'EngineeringKnowledge': ('.episodes', 'EngineeringKnowledge'),
    'EvidenceClaim': ('.episodes', 'EvidenceClaim'),
    'ExperienceOutcome': ('.episodes', 'ExperienceOutcome'),
    'KnowledgeKind': ('.episodes', 'KnowledgeKind'),
    'ValidationEvidence': ('.episodes', 'ValidationEvidence'),
    'VerificationLevel': ('.episodes', 'VerificationLevel'),
}

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value

__all__ = list(_EXPORTS)
