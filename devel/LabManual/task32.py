# balanced random cortex-like network - analysing network behaviour
# modifying the behavior of the network by altering the values of the mean weight of the excitatory and inhibitory connections
# increase from 0.1nA for the excitatory and input connections to 0.11nA
# increase from -0.4nA for the inhibitory connection to -0.44nA
# what we observe is that the overall spiking activity is reduced
import pyNN.spiNNaker as p
import pylab

# -----------------------------------------------------
mean_exc_wt = 0.11
mean_inh_wt = -0.44
# -----------------------------------------------------
# simulation setup
p.setup(timestep=0.1)
# -----------------------------------------------------
# network setup
n_neurons = 100
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
                                                                parameters=[mean_exc_wt, 0.1], boundaries=(0,10), constrain="redraw"),
                                   delays=p.RandomDistribution(distribution="normal",
                                                               parameters=[1.5, 0.75], boundaries=(0,10), constrain="redraw"))
p.Projection(pop_exc, pop_inh, conn, target="excitatory")
# create a self-connection for the excitatory population
conn = p.FixedProbabilityConnector(conn_prob,
                                   weights=p.RandomDistribution(distribution="normal",
                                                                parameters=[mean_exc_wt, 0.1], boundaries=(0,10), constrain="redraw"),
                                   delays=p.RandomDistribution(distribution="normal",
                                                               parameters=[1.5, 0.75], boundaries=(0,10), constrain="redraw"))
p.Projection(pop_exc, pop_exc, conn, target="excitatory")
# create an inhibitory connection from the inhibitory population to the excitatory population
conn = p.FixedProbabilityConnector(conn_prob,
                                   weights=p.RandomDistribution(distribution="normal",
                                                                parameters=[mean_inh_wt, 0.1], boundaries=(-10,0), constrain="redraw"),
                                   delays=p.RandomDistribution(distribution="normal",
                                                               parameters=[0.75, 0.375], boundaries=(-10,0), constrain="redraw"))
p.Projection(pop_inh, pop_exc, conn, target="inhibitory")
# create an inhibitory connection from the inhibitory to itself
conn = p.FixedProbabilityConnector(conn_prob,
                                   weights=p.RandomDistribution(distribution="normal",
                                                                parameters=[mean_inh_wt, 0.1], boundaries=(-10,0), constrain="redraw"),
                                   delays=p.RandomDistribution(distribution="normal",
                                                               parameters=[0.75, 0.375], boundaries=(-10,0), constrain="redraw"))
p.Projection(pop_inh, pop_inh, conn, target="inhibitory")
# setup special parameters for the neurons in the balanced network
pop_exc.initialize("v", p.RandomDistribution(distribution="uniform", parameters=[-65,-55]))
pop_inh.initialize("v", p.RandomDistribution(distribution="uniform", parameters=[-65,-55]))
# record the spikes in the excitatory population and run the simulation for 1+ seconds
# record the populations spike trains and the membrane voltages
pop_exc.record()
# to get the membrane potentials
pop_exc.record_v()
# -----------------------------------------------------
# run the simulation for 1.5 s
sim_time = 1500 # ms
p.run(sim_time)
# ---------
# plot the spike raster
# get spike trains
pylab.figure()
spikes = pop_exc.getSpikes() # this provides an array with [spike_id, spike_time]
spike_time = [i[1] for i in spikes]
spike_id = [i[0] for i in spikes]
pylab.plot(spike_time, spike_id,"o")
pylab.xlabel("Time (ms)")
pylab.ylabel("Neuron ID")

# -----------------------------------------------------
# plot the membrane voltages
pylab.figure()
memb_volt = pop_exc.get_v()
for nid in range(0, n_neurons):
    time = [i[1] for i in memb_volt if i[0] == nid]
    volt = [i[2] for i in memb_volt if i[0] == nid]
    pylab.subplot(n_neurons, 1, nid+1)
    pylab.plot(time, volt)
    pylab.axis([0,sim_time,-75,-45])

pylab.xlabel("Time(ms)")
pylab.ylabel("Membrane potentials")
pylab.show()