from GPU_AI import *
from MNIST import *
P = MNIST()
H = GPU_Herd(P.nb_sensors, P.nb_actors, 32, period=3)
#H = GPU_Herd(2, 2, 2, period=3)

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

mf = cl.mem_flags
definitions = defines(H.nb_sensors, H.nb_actors, H.nb_add_neurons, H.period, H.function.__name__)
#definitions = defines(1, 1, 1, 1, H.function.__name__)
function_code = C_to_string(H.function.__name__ + ".c")
AI_code = C_to_string("Kernel_AI.c")
code = definitions + function_code + AI_code + P.Kernel_code()
kernel_program = cl.Program(context, code).build()
#kernel_program.experience
# Start Evolving
Problem_inputs = P.Kernel_inputs(H.size*H.nb_tests)
Network_inputs = []
for member in H.members:
    Network_values = []
    for value in member.flatten():
        Network_values.append(value)
    Network_inputs.append(Network_values)
Kernel_inputs = []
for i in range(H.nb_tests*H.size):
    Kernel_inputs += Problem_inputs[i]
    Kernel_inputs += Network_inputs[i%(H.nb_tests*H.size)]
Kernel_inputs_buffer = cl.Buffer(context,
                                 mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=np.array(Kernel_inputs))
Kernel_inputs = np.array(Kernel_inputs)
score = np.zeros((H.size*H.nb_tests, ))
score_buffer = cl.Buffer(context, mf.WRITE_ONLY, score.nbytes)
kernel_program.experience(queue, Kernel_inputs.shape, None, Kernel_inputs_buffer, score_buffer)
cl.enqueue_copy(queue, score, score_buffer)
