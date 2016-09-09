# task 2.1 - synfire chain with random delays among the stimulus and the chain
import pyNN.spiNNaker as p
import pylab


# number of neurons in each population
chain_dim = 1
chain_size = 50
# -----------------------------------------------------
p.setup(timestep=1.0)
# -----------------------------------------------------
# input spike source spiking at 0ms
ninput = 1
spikeArray = {'spike_times': [[0]]}
stimulus = p.Population(ninput, p.SpikeSourceArray, spikeArray, label='stimulus')
# create the neural synfire chain, 100 chained neurons
chain_pops = [
    p.Population(chain_dim, p.IF_curr_exp, {}, label='chain_level_{}'.format(i))
    for i in range(chain_size)
]
# -----------------------------------------------------
# record the population
for pop in chain_pops:
    pop.record()
# -----------------------------------------------------
# connections within the population and with the stimulus
# different weight and delay from input to chain
# the connection delays are drawn from a random distribution using a uniform proability function
connector = p.OneToOneConnector(weights=5.0, delays=p.RandomDistribution(distribution='uniform', parameters=[1,15]))
for i in range(chain_size):
    p.Projection(chain_pops[i], chain_pops[(i + 1) % chain_size], connector)
p.Projection(stimulus, chain_pops[0], p.OneToOneConnector(weights=5.0, delays=1.0))
# -----------------------------------------------------
# run for 5 seconds and get spike raster
p.run(5000)
spikes = [pop.getSpikes() for pop in chain_pops]
p.end()
# -----------------------------------------------------
# plot and analysis
pylab.figure()
pylab.xlabel('Time (ms)')
pylab.ylabel('Neuron')
pylab.title('Spikes Sent By Chain')
offset = 0
for pop_spikes in spikes:
    pylab.plot(
        [i[1] for i in pop_spikes], [i[0] + offset for i in pop_spikes], "."
    )
    offset += chain_size
pylab.show()
