from ml.rf import est as base_est
from ml.mc import mc_params
from pmc.multicalibrator import MultiCalibrator

    
est = MultiCalibrator(
        estimator = base_est,
        **mc_params
    )
    

