typedef struct Network
{
    float values[NB_TOTAL_NEURONS];
    float bias[NB_TOTAL_NEURONS];
    float weights[NB_TOTAL_NEURONS][NB_TOTAL_NEURONS];
}
Network;

void iteration(Network *N)
{
    float values2[NB_TOTAL_NEURONS];
    int i;
    int j;
    for (i=0; i<NB_TOTAL_NEURONS; i++)
    {
        values2[i] = 0;
    }
    for (i=0; i<NB_TOTAL_NEURONS; i++)
    {
        for (j=0; j<NB_TOTAL_NEURONS; j++)
        {
            values2[j]+=N->weights[j][i]*(N->bias[i] + N->values[i]);
        }
    }
    for (i=0; i<NB_TOTAL_NEURONS; i++)
    {
        N->values[i] = FUNCTION(values2[i]);
    }
}

__kernel void reproduce(__global float *Kernel_inputs)
    {
    	
    }

