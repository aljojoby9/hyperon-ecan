from dataclasses import dataclass


@dataclass
class ECANParams:
    """Tunable constants. Names in angle brackets follow Iklé et al. 2009."""

    rent: float = 0.8
    wage: float = 1.0
    conj_decay: float = 0.35
    max_spread: float = 0.4
    focus_boundary: float = 8.0
    recent_max_sti: float = 40.0
    recent_min_sti: float = 0.0
    lti_forget_fraction: float = 0.05
    lti_forget_floor: float = 0.15
    neural_mix: float = 0.4
    neural_threshold: float = 0.42
    embedding_dim: int = 64
    sti_min: float = -20.0
    sti_max: float = 80.0
    bank_sti: float = 400.0
    bank_lti: float = 400.0
    lti_rent: float = 0.05
    lti_wage: float = 0.25
