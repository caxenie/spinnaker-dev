# tasks on simple IO within pyNN using a synfire neural network
# now we consider 2 synfire populations connected to the same spike injector
import spynnaker.pyNN as p
import spynnaker_external_devices_plugin.pyNN as ExternalDevices
# import to allow prefix type for the prefix eieio protocol
from spynnaker_external_devices_plugin.pyNN.connections\
    .spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
import pylab
import time
import random
from threading import Condition


# Create an initialisation method
def init_pop(label, n_neurons, run_time_ms, machine_timestep_ms):
    print "{} has {} neurons".format(label, n_neurons)
    print "Simulation will run for {}ms at {}ms timesteps".format(
        run_time_ms, machine_timestep_ms)


# Create a sender of packets for the synfire population
def send_input(label, sender):
    for neuron_id in range(0, n_neurons, 1):
        time.sleep(random.random() + 0.5)
        print_condition.acquire()
        print "Sending spike", neuron_id
        print_condition.release()
        sender.send_spike(label, neuron_id, send_full_keys=True)


# Create a receiver of live and update logic to fire the seconf neuron
# when the last neuron in the first synchain fires and does the same when
# the last neuron from the second synchain sets off some neuron id of the
# first synchain
def receive_spikes(label, time, neuron_ids):
    for neuron_id in neuron_ids:
        print_condition.acquire()
        print "Received spike at time", time, "from", label, "-", neuron_id
        print_condition.release()
        if neuron_id == 0 and label == "synchain1":
            live_spikes_connection.send_spike("synchain2", 0)
        elif neuron_id == 99 and label == "synchain2":
            live_spikes_connection.send_spike("synchain1", 1)

# initial call to set up the front end (pynn requirement)
p.setup(timestep=1.0)

# choose if using the c visualizer
use_c_visualiser = True

# neurons per population and the length of runtime in ms for the simulation,
# as well as the expected weight each spike will contain
n_neurons = 100
run_time = 10000
weight_to_spike = 2.0

# neural parameters of the LIF neurn model model used to respond to injected spikes.
# (cell params for a synfire chains)
cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 2.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }

# create synfire populations
synfire_chain_1 = p.Population(n_neurons, p.IF_curr_exp,
                                  cell_params_lif, label='synchain1')
synfire_chain_2 = p.Population(n_neurons, p.IF_curr_exp,
                                  cell_params_lif, label='synchain2')

# Create a single injection population feeding both synfire chains
injector = p.Population(
    n_neurons, ExternalDevices.SpikeInjector,
    {'port': 12346,  'virtual_key': 0x70000},
    label='spike_injector')

# Create a connection from the injector into the two synfire chains
p.Projection(injector, synfire_chain_1,
             p.FromListConnector([[0, 0, weight_to_spike, 3]]))
p.Projection(injector, synfire_chain_2,
             p.FromListConnector([[0, 0, weight_to_spike, 3]]))

# Synfire chain connections where each neuron is connected to its next neuron
# NOTE: there is no recurrent connection so that each chain stops once it
# reaches the end
loop = list()
for i in range(0, n_neurons - 1):
    loop.append((i, (i + 1) % n_neurons, weight_to_spike, 3))

p.Projection(synfire_chain_1, synfire_chain_1, p.FromListConnector(loop))
p.Projection(synfire_chain_2, synfire_chain_2, p.FromListConnector(loop))

# record spikes from the synfire chains so that we can read off valid results
# in a safe way afterwards, and verify the behavior
synfire_chain_1.record()
synfire_chain_2.record()

# Activate the sending of live spikes
ExternalDevices.activate_live_output_for(
    synfire_chain_1, database_notify_host="localhost",
    database_notify_port_num=19996)
ExternalDevices.activate_live_output_for(
    synfire_chain_2, database_notify_host="localhost",
    database_notify_port_num=19996)
#Activate the sending of spikes for the python reciever
ExternalDevices.activate_live_output_for(synfire_chain_1,
                                         database_notify_host="localhost",
                                         database_notify_port_num=19995,
                                         port=13333)
ExternalDevices.activate_live_output_for(synfire_chain_2,
                                         database_notify_host="localhost",
                                         database_notify_port_num=19995,
                                         port=13333)

# Create a condition to avoid overlapping prints
print_condition = Condition()

# Set up the live connection for sending spikes
live_spikes_connection = SpynnakerLiveSpikesConnection(
        receive_labels=None, local_port=19996,
        send_labels=["spike_injector"])

live_spikes_connection_recv = SpynnakerLiveSpikesConnection(
        receive_labels=["synchain1","synchain2"],
        local_port=19995,
        send_labels=["spike_injector"])

# check which connection we use
# Set up the live connection for sending spikes
# THIS SHOULD BE ADDED IN THE COLOR MAP
# Set up callbacks to occur when spikes are received
live_spikes_connection_recv.add_receive_callback(
    "synchain1", receive_spikes)
# Set up callbacks to occur when spikes are received
live_spikes_connection_recv.add_receive_callback(
    "synchain2", receive_spikes)

# Set up callbacks to occur at the start of simulation
live_spikes_connection.add_start_callback(
    "spike_injector", send_input)

# register socket addresses for the external system
p.register_database_notification_request(hostname="localhost",
                                         notify_port=19998,
                                         ack_port=19997)

# Run the simulation on spiNNaker
p.run(run_time)

if not use_c_visualiser:
    # Retrieve spikes from the synfire chain population
    spikes = synfire_chain.getSpikes()

    # If there are spikes, plot using matplotlib
    if len(spikes) != 0:
        pylab.figure()
        if len(spikes) != 0:
            pylab.plot([i[1] for i in spikes],
                       [i[0] for i in spikes], "b.")
        pylab.ylabel('neuron id')
        pylab.xlabel('Time (ms)')
        pylab.title('spikes')
        pylab.show()
    else:
        print "No spikes received"

# Clear data structures on spiNNaker to leave the machine in a clean state for
# future executions
p.end()