# Non Directional Neural Network

## What is it?
It's a neural network with only one layer, but this layer loops back onto
itself, just like how the brain does: multiple neurons called at the same
time and which call each other right after.

## What is it disigned for?
The goal with this architecture is to mimic the brain and see what happens.
I have some ideas on what type of data this architecture will be good for.
For exemple live problems like driving a car: The network is constantly fed
with informations and is constantly analysing them, what's more is that 
if the car see any danger it will be able to short cut it's process by
using the networks that are directly between the input and the output.
Antother exemple is when a problem involve some long/short term memory,
this architecture is able to provide memory cells that actually are neurons
that loops on themselves.

## How to use it?
The main program is called AI.py and it provides the user with four useful
classes : Problem, Herd, Network and TestBench.
The Problem class is the framework of any live problem so it's easier for the
user to code a new one, or to use it as an empty test problem.
The Herd class is a group of Network, it is this class that manages natural
selection among the Networks.
The Network class is the one doing all the computing by running the live
inputs fed from the Problem into itself.
The TestBench class is here to see how everything matchup by providing some
tests the user might want to perform.

First what you want to do is find/code a Problem you like and initialize it.
Then you want to make your Network learn on it, so typically you would use a
TestBench to do so.
Then you run the test nb°0 and your machine starts to learn!

For instance this can look like that car the Car.py problem

```python
P = Car(False, 8)
TB = TestBench(
    P,
    1, # nb_herds
    50, # nb_generations
    9, # nb_add_neurons
    1, # period
    500, # size
    0.02, # mutation_coefficient
    0.001, # mutation_amplitude
    20, # nb_tests
    slices=[5, 4],
    regions=[
        [False, False, False, False]
        [True, False, False, False]
        [False, True, False, False],
        [False, False, True, False],
    ]
)
TB.test(0)
```

Next what you can do is load_networking the saved network and see how it performs.

In this exemple it would look like that:

```python
N = load_network("Car_place_your_date_in_here")[0]
P = Car(True, 8)
P.experience(N)
```

Now if you gave a Network that works at last perfectly fine you can compile it using the compile command

```python
N.compile("My_executable")
```

And use it with streams of input and output

```zsh
./My_executable /dev/stdin /dev/stdout
```

Have a nice learning session!
