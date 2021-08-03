__kernel void experience(__global const float *input,
		__global int *score)
{
	// input values format
	// [number1, image1..., bias1..., weights1......., number2, ....]
	// Values initialisation
	int gid = get_global_id(0);
	int valuesSize = 1+NB_SENSORS+NB_TOTAL_NEURONS*(1+NB_TOTAL_NEURONS);
	int number = input[gid*valuesSize];
	Network N;
	for (int i = 0; i < NB_TOTAL_NEURONS; i++)
	{
		N.bias[i] = input[gid*valuesSize+1+NB_SENSORS+i];
		for (int j = 0; j < NB_TOTAL_NEURONS; i++)
		{
			N.weights[j][i] = input[
				gid*valuesSize+1+NB_SENSORS+(i+1)*NB_TOTAL_NEURONS+j];
    	}
	}
	for (int i = 0; i < NB_SENSORS; i++)
	{
		N.values[i] = input[gid*valuesSize+1+i];
	}
	// Network working...
	for (int i = 0; i < PERIOD; i++)
	{
		iteration(&N);
	}
	// Getting the output of the Network
	float max = -2;
	int indice = 0;
	for (int i = 0 ; i < NB_ACTORS; i++)
	{
		int indice_actor = i + NB_TOTAL_NEURONS - NB_ACTORS;
		if (N.values[indice_actor] > max)
    	{
			max = N.values[indice_actor];
			indice = indice_actor;
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

