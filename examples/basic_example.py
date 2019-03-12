# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import root_pandas

import templatefitter as tf


# In[3]:

ceph = "/ceph/welsch/moritz_data/"

signal = pd.read_pickle(ceph+"PI0_SIGNAL.pkl")
remaining = pd.read_pickle(ceph+"PI0_remaining.pkl")
xulnu = pd.read_pickle(ceph+"PI0_xulnu.pkl")
data = pd.read_pickle(ceph+"DATA.pkl")

data.head()


# ## Prepare Templates

# In[4]:


num_bins = 25
limits = (-1, 3)


# ### $e$ channel

# In[5]:


e_sig = signal.query("SignalChannel==1.")
e_remaining = remaining.query("SignalChannel==1.")
e_xulnu = xulnu.query("SignalChannel==1.")
e_data = data.query("SignalChannel==1.")

e_sig_eff = sum(e_sig.weight)/sum(signal.weight)
e_remaining_eff = sum(e_remaining.weight)/sum(remaining.weight)
e_xulnu_eff = sum(e_xulnu.weight)/sum(xulnu.weight)

print(e_sig_eff, e_remaining_eff, e_xulnu_eff)

e_hsig = tf.histograms.Hist1d(bins=num_bins, range=limits, data=e_sig.missingMass, weights=e_sig.weight)
e_hremaining = tf.histograms.Hist1d(bins=num_bins, range=limits, data=e_remaining.missingMass,  weights=e_remaining.weight)
e_hxulnu = tf.histograms.Hist1d(bins=num_bins, range=limits, data=e_xulnu.missingMass,  weights=e_xulnu.weight)

e_tsig = tf.templates.Template1d("Signal", "missingMass", e_hsig, color="cornflowerblue")
e_tremaining = tf.templates.Template1d("Remaining", "missingMass", e_hremaining, color="indianred")
e_txulnu = tf.templates.Template1d("Xulnu", "missingMass", e_hxulnu, color="khaki")


# In[6]:


for template in [e_tsig,e_tremaining, e_txulnu]:
    fig, axis = plt.subplots(1,1)
    template.plot_on(axis)


# ### $\mu$ channel

# In[7]:


mu_sig = signal.query("SignalChannel==2.")
mu_remaining = remaining.query("SignalChannel==2.")
mu_xulnu = xulnu.query("SignalChannel==2.")
mu_data = data.query("SignalChannel==2.")

mu_sig_eff = sum(mu_sig.weight)/sum(signal.weight)
mu_remaining_eff = sum(mu_remaining.weight)/sum(remaining.weight)
mu_xulnu_eff = sum(mu_xulnu.weight)/sum(xulnu.weight)

print(mu_sig_eff, mu_remaining_eff, mu_xulnu_eff)

mu_hsig = tf.histograms.Hist1d(bins=num_bins, range=limits, data=mu_sig.missingMass, weights=mu_sig.weight)
mu_hremaining = tf.histograms.Hist1d(bins=num_bins, range=limits, data=mu_remaining.missingMass, weights=mu_remaining.weight)
mu_hxulnu = tf.histograms.Hist1d(bins=num_bins, range=limits, data=mu_xulnu.missingMass, weights=mu_xulnu.weight)

mu_tsig = tf.templates.Template1d("Signal", "missingMass", mu_hsig, color="cornflowerblue")
mu_tremaining = tf.templates.Template1d("Remaining", "missingMass", mu_hremaining, color="indianred")
mu_txulnu = tf.templates.Template1d("Xulnu", "missingMass", mu_hxulnu, color="khaki")


# In[8]:


for template in [mu_tsig, mu_tremaining, mu_txulnu]:
    fig, axis = plt.subplots(1,1)
    template.plot_on(axis)


# ## MultiChannelTemplate

# In[9]:


mct = tf.templates.MultiChannelTemplate()

mct.define_channel("e", num_bins, limits)
mct.define_channel("mu", num_bins, limits)
mct.define_process("signal")
mct.define_process("remaining")
mct.define_process("xulnu")


mct.add_template("e", "xulnu", e_txulnu, e_xulnu_eff)
mct.add_template("e", "remaining", e_tremaining, e_remaining_eff)
mct.add_template("e", "signal", e_tsig, e_sig_eff)

mct.add_template("mu", "xulnu", mu_txulnu, mu_xulnu_eff)
mct.add_template("mu", "remaining", mu_tremaining,mu_remaining_eff)
mct.add_template("mu", "signal", mu_tsig, mu_sig_eff)

mct.add_data(e=tf.histograms.Hist1d(num_bins, limits, data=e_data.missingMass))
mct.add_data(mu=tf.histograms.Hist1d(num_bins, limits, data=mu_data.missingMass))

e_tsig.yield_param


# In[10]:


for channel in mct.channels.values():
    fig, axis = plt.subplots(1,1, figsize=(8,8))
    channel.plot_stacked_on(axis)


# In[11]:


fitter = tf.TemplateFitter(mct, "iminuit")


# In[12]:


fitter.do_fit(update_templates=True)


# In[18]:


print(fitter.get_significance("signal"))


# In[14]:


for channel in mct.channels.values():
    fig, axis = plt.subplots(1,1, figsize=(8,8))
    channel.plot_stacked_on(axis)
    axis.legend()


# In[15]:


sig_yield, nll_profile, hesse = fitter.profile("signal_yield", num_cpu=25)
fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=200)
ax.plot(sig_yield, nll_profile, label="profile")
ax.plot(sig_yield, hesse, label="hesse approx.")
plt.show()

