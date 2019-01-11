import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from templatefitter import *

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



PI0_signal = pd.read_pickle("/ceph/welsch/pi0_samples/PI0_SIGNAL.pkl")
PI0_remaining = pd.read_pickle("/ceph/welsch/pi0_samples/PI0_remaining.pkl")
PI0_xulnu = pd.read_pickle("/ceph/welsch/pi0_samples/PI0_xulnu.pkl")
PI0_data= pd.read_pickle("/ceph/welsch/pi0_samples/DATA.pkl")

tc = AdvancedCompositeTemplate("missingMass", 30, (-1, 3.))
tc.create_template("xulnu", PI0_xulnu)
tc.create_template("remaining", PI0_remaining)
tc.create_template("signal", PI0_signal)

hdata = Histogram(30, (-1, 3.))
hdata.fill(PI0_data.missingMass)

tc.bin_fractions(np.zeros(30*3))


fig, ax = plt.subplots(1,1, figsize=(10,10))

tc.reset_yield_values()
print(tc["signal"].yield_value)
print(tc["signal"].rel_uncertainties)
print(tc["signal"].uncertainties)

tc.plot_on(ax)
ax.legend(loc=2, fontsize=20)
ax.errorbar(hdata.bin_mids, hdata.bin_counts, yerr=hdata.bin_errors, ls="", marker="o", color='black')

hfake_data = Histogram(30, (-1, 3.))
hfake_data.bin_counts = tc.generate_asimov_dataset()
fitter = TemplateFitter(hdata, tc, AdvancedPoissonNegativeLogLikelihood)
params = fitter.do_fit(get_hesse=False)
print(params)


print(params.values)
print(params.errors)

tc.update_parameters(params.values)
fig, ax = plt.subplots(1,1, figsize=(10,10))

print(tc["signal"].yield_value)
print(tc["signal"].rel_uncertainties)
print(tc["signal"].uncertainties)

tc.plot_on(ax)
ax.legend(loc=2, fontsize=20)
ax.errorbar(hdata.bin_mids, hdata.bin_counts, yerr=hdata.bin_errors, ls="", marker="o", color='black')
