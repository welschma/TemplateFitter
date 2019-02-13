Tutorial
========

Basic Example
#############
Basic example on how to use the TemplateFitter package

.. code-block:: python

    import pandas as pd
    import matplotlib.pyplot as plt
    import templatefitter as tf

    # Loading the DataFrames
    source_path = "/some/absolute/path"
    sig = pd.read_pickle(os.path.join(source_path, "PI0_signal.pkl"))
    remaining = pd.read_pickle(os.path.join(source_path, "PI0_remaining.pkl"))
    xulnu = pd.read_pickle(os.path.join(source_path, "PI0_xulnu.pkl"))
    data =  pd.read_pickle(os.path.join(source_path, "PI0_data.pkl"))

    # Create the template
    nbins = 25
    limits = (-1.5, 3)
    st = tf.StackedTemplate(name="PI0_ALL", variable="missinMass", num_bins=nbins, limits=limits)
    # add templates to the templates
    # the order only matters in terms of the plotting order and the order how
    # parameters are handled internally
    st.create_template("xulnu", xulnu)
    st.create_template("remaining", remaining)
    st.create_template("signal", sig)

    # create a binned dataset
    data = tf.Hist1d(nbins, limits, data.missingMass)

    # you could also create a toy or asimov dataset from the current template
    # data = st.generate_toy_dataset()
    # data = st.generate_asimov_dataset(integer_values=False)

    # plot the pre fit distribution and the dataset
    # note: only the histograms are plotted, if you want to have a nicer plot
    # with axis titles, you have to set them yourself
    fig, ax = plt.subplots(1, 1, figsize=(8,8), dpi=200)
    st.plot_on(ax)
    ax.errorbar(data.bin_mids, data.bin_counts, yerr=data.bin_errors, ls="", marker=".", color="black")

    # perform the fit, you should get the fit result for the parameters as output in your terminal
    fitter = tf.TemplateFitter(data, st, "iminuit")
    fitter.do_fit(verbose=True)

    # plot the fit result
    fig, ax = plt.subplots(1, 1, figsize=(8,8), dpi=200)
    st.plot_on(ax)
    ax.errorbar(data.bin_mids, data.bin_counts, yerr=data.bin_errors, ls="", marker=".", color="black")

    # create a profile likelihood plot for the signal yield parameter
    points, profile, hesse = fitter.profile("signal_yield")
    fig, ax = plt.subplots(1, 1, figsize=(8,8), dpi=200)
    ax.plot(points, profile, label="Profile NLL")
    ax.plot(points, hesse, label="Hesse approx.")

    # create a toy study to perform linearity tests for yield paramters or for studying pull distributions
    toys = tf.ToyStudy(st, "iminuit")
    result = toys.do_linearity_test("signal", (0, 1000), n_points=3, n_exp=30)
    fig, ax = plt.subplots(1, 1, figsize=(8,8), dpi=200)
    ax.errorbar(x=result[0], y=result[1], yerr=result[2], ls="", marker=".")

