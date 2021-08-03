__kernel void experience(__global const float *input,
		__global int *score)
{
	// input values format
	// [number1, image1..., bias1..., weights1......., number2, ....]
	// Values initialisation
	int gid = get_global_id(0);
	int valuesSize = 1+NB_SENSORS+NB_TOTAL_NEURONS*(1+NB_TOTAL_NEURONS);
	int number = input[gid*valuesSize];
	// Network = [bias... weights... values]
	float Network[NB_TOTAL_NEURONS*(1+NB_TOTAL_NEURONS)];
	for (int i = 0; i < NB_TOTAL_NEURONS; i++)
	{
		Network[i] = input[gid*valuesSize+1+NB_SENSORS+i];
		for (int j = 0; j < NB_TOTAL_NEURONS; i++)
		{
			Network[NB_TOTAL_NEURONS * (1 + i) + j] = input[
				gid*valuesSize+1+NB_SENSORS+(i+1)*NB_TOTAL_NEURONS+j];
    	}
	}
	for (int i = 0; i < NB_SENSORS; i++)
	{
		Network[NB_TOTAL_NEURONS + i] = input[gid*valuesSize+1+i];
	}
	// Network working...
	for (int i = 0; i < PERIOD; i++)
	{
		iteration(Network);
	}
	// Getting the output of the Network
	float max = -2;
	int indice = 0;
	for (int i = 0 ; i < NB_ACTORS; i++)
	{
		if (Network[NB_TOTAL_NEURONS + i] > max)
    	{
			max = Network[NB_TOTAL_NEURONS + i];
			indice = i;
    	}
	}
	// Evaluation
	if (number == indice)
	{
		score[gid] = 1;
	}
	else
	{
		score[gid] = 0;
	}
}

