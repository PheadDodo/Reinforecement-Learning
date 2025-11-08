import numpy as np

class ContextualNewsEnv:
    """
    Contextual bandit for news with interpretable features, incl. age.
    Reward ~ Bernoulli(sigmoid(theta[a] · x)).
    """
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        # 6 human-readable features (last one is age_z = (age - 40)/15)
        self.feature_names = [
            "likes_politics",
            "sports_fan",
            "techie",
            "mobile_user",
            "morning_reader",
            "age_z",             # standardized age (0 ≈ 40y, +1 ≈ 55y, -1 ≈ 25y)
        ]
        self.d = len(self.feature_names)
        self.arm_names = ["Politics", "Sports", "Tech", "Lifestyle"]
        self.k = len(self.arm_names)

        # True arm parameters (rows=arms, cols=features); include age effects
        self.theta = np.array([
            [ 1.6,  0.2,  0.1,  0.2,  0.7,  0.4],   # Politics prefers older & morning readers
            [ 0.1,  1.8,  0.1,  0.7,  0.2, -0.1],   # Sports slightly skew younger/mobile
            [ 0.0,  0.1,  1.9, -0.1, -0.2, -0.2],   # Tech slightly skew younger
            [ 0.3,  0.2,  0.2,  1.0,  0.8,  0.0],   # Lifestyle mostly device/time driven
        ], dtype=float)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def _sample_age_z(self):
        """
        Sample an age in years, then standardize:
            age_z = (age - 40) / 15
        (≈ N(40, 12^2) clipped to [18, 80] for realism)
        """
        age = float(np.clip(self.rng.normal(40, 12), 18, 80))
        return (age - 40.0) / 15.0

    def sample_context(self):
        """
        Sample an interpretable user:
        pick a coarse segment, then add noise; append standardized age.
        """
        seg = self.rng.choice(["politics", "sports", "tech", "on_the_go", "morning_person"])
        x = np.zeros(self.d)
        if seg == "politics":
            core = np.array([1.6, 0.2, 0.2, 0.3, 0.9])
        elif seg == "sports":
            core = np.array([0.2, 1.8, 0.2, 1.0, 0.3])
        elif seg == "tech":
            core = np.array([0.2, 0.2, 1.9, 0.3, 0.2])
        elif seg == "on_the_go":
            core = np.array([0.4, 0.9, 0.5, 1.8, 0.7])
        else:  # morning_person
            core = np.array([0.8, 0.3, 0.2, 0.6, 1.9])
        core = core + self.rng.normal(0, 0.2, size=5)
        age_z = self._sample_age_z()
        x[:5] = core
        x[5] = age_z
        return x

    def click_prob(self, arm, x):
        return float(self._sigmoid(self.theta[arm] @ x))

    def click(self, arm, x):
        """Return (reward, true_click_prob)."""
        p = self.click_prob(arm, x)
        r = int(self.rng.random() < p)
        return r, p












