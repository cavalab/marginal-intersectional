from ml.rf import est as base_est
from fomo.estimator import FomoClassifier
from fomo.problem import (
    BasicProblem, MLPProblem, LinearProblem, InterLinearProblem
)

from ml.fomo_shared_args import shared_args
shared_args.update(dict(estimator = base_est))

est_MLP = FomoClassifier(**shared_args, problem_type=MLPProblem)

est_lin = FomoClassifier(**shared_args, problem_type=LinearProblem)

est_interlin = FomoClassifier(**shared_args, problem_type=InterLinearProblem)