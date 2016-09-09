# introducing a network using STDP learning rules
import pyNN.spiNNaker as p
import pylab

p.setup(timestep=1.0)
# -----------------------------------------------------
# create a 2 single neuron populations connected through a STDP synapse
n_neurons = 10
init_weights = 0.0
# create the populations
pre_pop = p.Population(n_neurons, p.IF_curr_exp, {}, label="pre_population")
post_pop = p.Population(n_neurons, p.IF_curr_exp, {}, label="post_population")
# each population gets stimulation from separate spike with different spike times
pre_pop_stim = p.Population(n_neurons, p.SpikeSourceArray, {"spike_times": range(0, 100, 2)}, label="pre_stim")
post_pop_stim = p.Population(n_neurons, p.SpikeSourceArray, {"spike_times": range(2, 100, 2)}, label="post_stim")
# -----------------------------------------------------
# setup the connectivity
p.Projection(pre_pop_stim, pre_pop, p.OneToOneConnector(weights=5.0))
p.Projection(post_pop_stim, post_pop, p.OneToOneConnector(weights=5.0))
# -----------------------------------------------------
# parametrize the STDP rule
timing_rule = p.SpikePairRule(tau_minus=2, tau_plus=2)
weight_rule = p.AdditiveWeightDependence(w_max=5.0, w_min=0.1, A_plus=0.25, A_minus=0.25)
stdp_model = p.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule)
# connect the two populations via the STDP connection
stdp_projection = p.Projection(pre_pop, post_pop, p.OneToOneConnector(weights=init_weights, delays=5.0),
                               synapse_dynamics=p.SynapseDynamics(slow=stdp_model))
# -----------------------------------------------------
# training
training = p.Population(n_neurons, p.SpikeSourceArray, {"spike_times": range(0, 100, 2)}, label="training_data")
p.Projection(training, pre_pop, p.OneToOneConnector(weights=5.0, delays=1.0))
p.Projection(training, post_pop, p.OneToOneConnector(weights=5.0, delays=10.0))
# -----------------------------------------------------
# run the model from some ms and collect spikes
# record population
pre_pop.record()
post_pop.record()
r_time = 100
p.run(r_time)
presp = pre_pop.getSpikes()
postsp = post_pop.getSpikes()
# print the weights
print stdp_projection.getWeights()
# -----------------------------------------------------
# plot the spike raster
pylab.figure()
pylab.xlim((0,r_time))
pylab.plot([i[1] for i in presp],[i[0] for i in presp],"bo")
pylab.plot([i[1] for i in postsp],[i[0] for i in postsp],"ro")
pylab.xlabel("Time (ms)")
pylab.ylabel("Spikes")
pylab.show()

