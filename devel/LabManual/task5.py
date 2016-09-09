# plot an STDP curve showing how the weight change varies with timing between spikes
import pyNN.spiNNaker as p
import pylab
import numpy as np

# -----------------------------------------------------
# simulation params
p.setup(timestep=1.0)
# -----------------------------------------------------
# parametrize neural model
n_neurons = 10
# create a presynaptic population and a stimulating spike source
pre_pop = p.Population(n_neurons, p.IF_curr_exp,{},label="presynaptic_population")
pre_stim = p.Population(n_neurons,
                        p.SpikeSourceArray,# each neurons spikes twice 200ms apart
                        {"spike_times": [[x, y] for x in range(0,100, 1) for y in range(200, 300, 1)]},
                        label="presynaptic_spike_train")
# create a postsynaptic population and a stimulating spike source
post_pop = p.Population(n_neurons, p.IF_curr_exp,{},label="postsynaptic_population")
post_stim = p.Population(n_neurons,
                        p.SpikeSourceArray,# each neuron spikes at 50ms
                        {"spike_times": np.tile(50, n_neurons)},
                        label="postsynaptic_spike_train")
# connect the input spike trains for both pre- and post-synaptic populations
p.Projection(pre_stim, pre_pop, p.OneToOneConnector(weights=0.5, delays=0.1))
p.Projection(post_stim, post_pop, p.OneToOneConnector(weights=0.5, delays=0.1))
# connect the pre and post populations through a STDP synaptic connectivity with
# initial weight 0.5 and min_w = 0 and max_w = 1
timing_rule = p.SpikePairRule(tau_minus=20.0, tau_plus=20.0)
weight_rule = p.AdditiveWeightDependence(w_max=1, w_min=0, A_plus=0.5, A_minus=0.5)
stdp_model = p.STDPMechanism(timing_dependence=timing_rule,
                             weight_dependence=weight_rule)
# connect the populations with the STDP linkage
stdp_projection = p.Projection(pre_pop, post_pop,
                               p.OneToOneConnector(weights=0.5, delays=1.0),
                               synapse_dynamics=p.SynapseDynamics(slow=stdp_model))
# -------------------------------------------------------
# training
training = p.Population(n_neurons, p.SpikeSourceArray, {"spike_times": range(0, 300, 2)}, label="training_data")
p.Projection(training, pre_pop, p.OneToOneConnector(weights=5.0, delays=1.0))
p.Projection(training, post_pop, p.OneToOneConnector(weights=5.0, delays=10.0))
# -----------------------------------------------------
# record the populations
pre_pop.record()
post_pop.record()
# simulation params
r_time = 400 # ms
p.run(r_time)
# get the generated spike trains along with the weights
pre_spk = pre_pop.getSpikes()
post_spk = post_pop.getSpikes()
stdp_weights = stdp_projection.getWeights()
# -----------------------------------------------------
# plot the spike raster
pylab.figure()
pylab.xlim((0,r_time))
pylab.plot([i[1] for i in pre_spk],[i[0] for i in pre_spk],"bo")
pylab.plot([i[1] for i in post_spk],[i[0] for i in post_spk],"ro")
pylab.xlabel("Time (ms)")
pylab.ylabel("Spikes")
pylab.show()