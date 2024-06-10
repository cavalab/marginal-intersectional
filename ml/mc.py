from pmc.auditor import Auditor

mc_params = dict(
    auditor_type=Auditor(),
    eta=0.25,
    metric='MC',
    max_iters=10**5,
    verbosity=0,
    n_bins=10,
    split=0.5
)
