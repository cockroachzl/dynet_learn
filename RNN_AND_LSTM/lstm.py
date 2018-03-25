import dynet as dy

print(dy.__file__)
print(dir(dy))
print(len(dir(dy)))
print('ParameterCollection' in dir(dy))

pc = dy.ParameterCollection()
NUM_LAYERS=2
INPUT_DIM=50
HIDDEN_DIM=10

builder = dy.LSTMBuilder(NUM_LAYERS, INPUT_DIM, HIDDEN_DIM, pc)
s0 = builder.initial_state()
x1 = dy.vecInput(INPUT_DIM)
s1=s0.add_input(x1)
y1 = s1.output()
print(y1.npvalue().shape)

#dy.print_graphviz()
