float sigmoid(float x)
{
    float r;
    r = -1 + (2/(1+exp(-x)));
    return r;
}

