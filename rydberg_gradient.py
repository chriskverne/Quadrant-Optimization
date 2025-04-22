import numpy as np
import pulser
from matplotlib import pyplot as plt


device = pulser.AnalogDevice

# Create a single atom in position 0,0
register = pulser.Register.from_coordinates([(0, 0)], prefix="q")
#register.draw()

# Checks that quantum Register matches device we picked (we only need to transition from |0| to |1|)
sequence = pulser.Sequence(register, device)
sequence.declare_channel("rydberg_global", "rydberg_global")
#print("The states used in the computation are", sequence.get_addressed_states())

# laser pulse
# 1000 ns, rabi = pi rad / micro second, detuning = 0, phase = 0
pi_pulse = pulser.Pulse.ConstantPulse(1000, np.pi, 0, 0)
sequence.add(pi_pulse, "rydberg_global")
#sequence.draw(mode="input") # Prints the lazer pulse sent

# Run pulse on backend
backend = pulser.backends.QutipBackend(sequence)
result = backend.run()
print(result.sample_final_state(1000))

