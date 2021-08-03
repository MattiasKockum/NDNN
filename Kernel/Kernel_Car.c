void matmul(int size_1_a, int size_1_b, double **mat1,
    int size_2_a, int size_2_b, double **mat2, double **matRes)
{
    assert size_1_b == size_2_a;
    double res;
    for (int i = 0; i < size_1_a, i++)
    {
        for (int j = 0; j < size_2_b, j++)
        {
            res = 0;
            for (int k = 0; k < size_1_b; k++)
            {
                res += mat1[i][k] * mat2[k][j];
            }
            matRes[i][j] = res;
        }
    }
}

void rotation(double **Vector2D, double angle, double **Vector2DRes)
{
    double rotation_matrice[2][2];
    rotation_matrice[0][0] = cos(angle);
    rotation_matrice[0][1] = -sin(angle);
    rotation_matrice[1][0] = sin(angle);
    rotation_matrice[1][1] = cos(angle);
    matmul(2, 2, rotation_matrice, 2, 1, Vector2D, Vector2DRes);
}
