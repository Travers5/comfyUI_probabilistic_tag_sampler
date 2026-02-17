# v0.01.00

from .probabilistic_tag_sampler import ProbabilisticTagSampler

NODE_CLASS_MAPPINGS = {
    "ProbabilisticTagSampler": ProbabilisticTagSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProbabilisticTagSampler": "Probabilistic Tag Sampler (Text)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
