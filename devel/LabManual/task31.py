# balanced random cortex-like network
import pyNN.spiNNaker as p
import pylab

# -----------------------------------------------------
# simulation setup
p.setup(timestep=0.1)
# -----------------------------------------------------
# network setup
n_neurons = 1000
# define the size of the 2 populations, 80% excitatory and 20% inhibitory
exc_size = int(round(n_neurons*0.8))
inh_size = int(round(n_neurons*0.2))
# create the neural populations in the network
pop_exc = p.Population(exc_size, p.IF_curr_exp, {}, label="excitatory_pop")
pop_inh = p.Population(inh_size, p.IF_curr_exp, {}, label="inhibitory_pop")
# create the input stimuli
exc_in = p.Population(exc_size, p.SpikeSourcePoisson, {"rate": 1000}, label="excitatory_in")
inh_in = p.Population(inh_size, p.SpikeSourcePoisson, {"rate": 1000}, label="inhibitory_in")
# connect the input population with Poisson spike distribution to the excitatory population
# with a weight from a 0.1nA current and a delay of 1ms using an excitatory connection
p.Projection(exc_in, pop_exc, p.OneToOneConnector(weights=0.1, delays=1), target="excitatory")
# connect the input population with Poisson spike distribution to the inhibitory population
# with a weight from a 0.1nA current and a delay of 1ms using an excitatory connection
p.Projection(inh_in, pop_inh, p.OneToOneConnector(weights=0.1, delays=1), target="excitatory")
# connect the excitatory and inhibitory populations
conn_prob = 0.1 # 10% of the neurons
conn = p.FixedProbabilityConnector(conn_prob,
                                   weights=p.RandomDistribution(distribution="normal",
                                                                parameters=[0.1, 0.1], boundaries=(0,10), constrain="redraw"),
                                   delays=p.RandomDistribution(distribution="normal",
                                                               parameters=[1.5, 0.75], boundaries=(0,10), constrain="redraw"))
p.Projection(pop_exc, pop_inh, conn, target="excitatory")
# create a self-connection for the excitatory population
conn = p.FixedProbabilityConnector(conn_prob,
                                   weights=p.RandomDistribution(distribution="normal",
                                                                parameters=[0.1, 0.1], boundaries=(0,10), constrain="redraw"),
                                   delays=p.RandomDistribution(distribution="normal",
                                                               parameters=[1.5, 0.75], boundaries=(0,10), constrain="redraw"))
p.Projection(pop_exc, pop_exc, conn, target="excitatory")
# create an inhibitory connection from the inhibitory population to the excitatory population
conn = p.FixedProbabilityConnector(conn_prob,
                                   weights=p.RandomDistribution(distribution="normal",
                                                                parameters=[-0.4, 0.1], boundaries=(-10,0), constrain="redraw"),
                                   delays=p.RandomDistribution(distribution="normal",
                                                               parameters=[0.75, 0.375], boundaries=(-10,0), constrain="redraw"))
p.Projection(pop_inh, pop_exc, conn, target="inhibitory")
# create an inhibitory connection from the inhibitory to itself
conn = p.FixedProbabilityConnector(conn_prob,
                                   weights=p.RandomDistribution(distribution="normal",
                                                                parameters=[-0.4, 0.1], boundaries=(-10,0), constrain="redraw"),
                                   delays=p.RandomDistribution(distribution="normal",
                                                               parameters=[0.75, 0.375], boundaries=(-10,0), constrain="redraw"))
p.Projection(pop_inh, pop_inh, conn, target="inhibitory")
# setup special parameters for the neurons in the balanced network
pop_exc.initialize("v", p.RandomDistribution(distribution="uniform", parameters=[-65,-55]))
pop_inh.initialize("v", p.RandomDistribution(distribution="uniform", parameters=[-65,-55]))
# record the spikes in the excitatory population and run the simulation for 1+ seconds
# record the populations spike trains and the membrane voltages
pop_exc.record()
# -----------------------------------------------------
# run the simulation for 1.5 s
sim_time = 1500 # ms
p.run(sim_time)
# ---------
# plot the spike raster
# get spike trains
spikes = pop_exc.getSpikes() # this provides an array with [spike_id, spike_time]
spike_time = [i[1] for i in spikes]
spike_id = [i[0] for i in spikes]
pylab.plot(spike_time, spike_id,"o")
pylab.xlabel("Time (ms)")
pylab.ylabel("Neuron ID")
pylab.show()