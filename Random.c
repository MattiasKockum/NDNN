void randomNext(double *seed)
{
	*seed = sin(1/(*seed%0.1));
}
