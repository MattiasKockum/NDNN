void iteration(float * Network)
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
            values2[j]+=Network[NB_TOTAL_NEURONS * (1 + i) + j]
				*(Network[i]
						+ Network[NB_TOTAL_NEURONS*(1+NB_TOTAL_NEURONS) + i]
				);
        }
    }
    for (i=0; i<NB_TOTAL_NEURONS; i++)
    {
        Network[NB_TOTAL_NEURONS + i] = FUNCTION(values2[i]);
    }
}

__kernel void reproduce(__global float *Kernel_inputs)
    {
    	
    }

