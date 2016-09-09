# task 1.2 - simple neural network
import pyNN.spiNNaker as p
import pylab

p.setup(timestep=1.0)
# -----------------------------------------------------
# input spike source
input_source = p.Population(2, p.SpikeSourceArray, {'spike_times' : [[0],[1]]}, label="spike_generator")
# neural population to project to
# adding the tau_syn_E to 1.0 (default is 5) the neurons do not spike anymore and the neurons do not reach the threshold voltage
pop = p.Population(2, p.IF_curr_exp, {'tau_syn_E': 5.0}, label="neural_population")
# -----------------------------------------------------
# create the projection
input_proj = p.Projection(input_source, pop, p.OneToOneConnector(weights=10.0, delays=2), target="excitatory")
# -----------------------------------------------------
# record the populations spike trains and the membrane voltages
pop.record()
pop.record_v()
# -----------------------------------------------------
# run the simulation for 10 ms
p.run(10)
# -----------------------------------------------------
# # get the voltages as a vector of [nid, ntime, nvoltage]
v = pop.get_v()
# -----------------------------------------------------
# # plotting utils
# # first plot the voltages for the first neuron
time = [i[1] for i in v if i[0] == 0]
membrane_voltage = [i[2] for i in v if i[0] == 0]
pylab.subplot(3, 1, 1)
pylab.plot(time, membrane_voltage)
pylab.xlabel("Time(ms)")
pylab.ylabel("Mem volt neuron 1")
pylab.axis([0,10,-75,-45])
# -----------------------------------------------------
time = [i[1] for i in v if i[0] == 1]
membrane_voltage = [i[2] for i in v if i[0] == 1]
pylab.subplot(3, 1, 2)
pylab.plot(time, membrane_voltage)
pylab.xlabel("Time(ms)")
pylab.ylabel("Mem volt neuron 2")
pylab.axis([0,10,-75,-45])
# -----------------------------------------------------
# plot the spike raster
spikes = pop.getSpikes()
spike_time = [i[1] for i in spikes]
spike_id = [i[0] for i in spikes]
pylab.subplot(3, 1, 3)
pylab.plot(spike_time, spike_id,"o")
pylab.xlabel("Time (ms)")
pylab.ylabel("Neuron ID")
pylab.axis([0,10,-1, 1])
pylab.show()
