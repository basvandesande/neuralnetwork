namespace NeuralNetwork
{
    public class Matrix
    {
        private readonly double[,] _data;
        
        public int Rows => _data.GetLength(0);
        
        public int Cols => _data.GetLength(1);
        
        public double this[int row, int col]
        {
            get => _data[row, col];
            set => _data[row, col] = value;
        }

        public Matrix()
        {
            _data = new double[1, 1];  
        }

        public Matrix(int rows, int cols)
        {
            _data = new double[rows, cols];
        }

        public Matrix (double[] array)
        {
            _data = new double[1, array.Length];
            for (int i = 0; i < array.Length; i++)
            {
                _data[0, i] = array[i];
            }
        }

        public static Matrix operator +(Matrix matrix1, Matrix matrix2)
        {
            if (matrix1.Rows != matrix2.Rows || matrix1.Cols != matrix2.Cols)
                throw new Exception("Matrices are not compatible");

            var output = new Matrix(matrix1.Rows, matrix1.Cols);
            for (int row = 0; row < matrix1.Rows; row++)
            {
                for (int col = 0; col < matrix1.Cols; col++)
                {
                    output[row, col] = matrix1[row, col] + matrix2[row, col];
                }
            }
            return output;
        }

        public static Matrix operator +(Matrix matrix, double value)
        {
            var output = new Matrix(matrix.Rows, matrix.Cols);
            for (int row = 0; row < matrix.Rows; row++)
            {
                for (int col = 0; col < matrix.Cols; col++)
                {
                    output[row, col] = matrix[row, col] + value;
                }
            }
            return output;
        }

        public static Matrix operator -(Matrix matrix1, Matrix matrix2)
        {
            if (matrix1.Rows != matrix2.Rows || matrix1.Cols != matrix2.Cols)
                throw new Exception("Matrices are not compatible");

            var output = new Matrix(matrix1.Rows, matrix1.Cols);
            for (int row = 0; row < matrix1.Rows; row++)
            {
                for (int col = 0; col < matrix1.Cols; col++)
                {
                    output[row, col] = matrix1[row, col] - matrix2[row, col];
                }
            }
            return output;
        }

        public static Matrix operator *(Matrix matrix1, Matrix matrix2)
        {
            // In order to multiply 2 matrices, the number of cols in matrix1 == number of rows in matrix2
            if (matrix1.Cols != matrix2.Rows)
                throw new Exception("Matrices are not compatible for multiplication.");

            var output = new Matrix(matrix1.Rows, matrix2.Cols);
            for (int i = 0; i < matrix1.Rows; i++)
            {
                for (int j = 0; j < matrix2.Cols; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < matrix1.Cols; k++)
                    {
                        sum += matrix1[i, k] * matrix2[k, j];
                    }
                    output[i, j] = sum;
                }
            }
            return output;
        }
                       
        public static Matrix operator *(Matrix matrix, double value)
        {
            var output = new Matrix(matrix.Rows, matrix.Cols);
            for (int row = 0; row < matrix.Rows; row++)
            {
                for (int col = 0; col < matrix.Cols; col++)
                {
                    output[row, col] = matrix[row, col] * value;
                }
            }
            return output;
        }
              
        public Matrix ApplyActivator(Func<double, double> activator)
        {
            var output = new Matrix(this.Rows, this.Cols);
            for (int row = 0; row < this.Rows; row++)
            {
                for (int col = 0; col < this.Cols; col++)
                {
                    output[row, col] = activator(this[row, col]);
                }
            }
            return output;
        }

        public Matrix Transpose()
        {
            var output = new Matrix(this.Cols, this.Rows);
            int row = 0;
            for (int col = 0; col < this.Cols; col++) 
            {
                double[] column = GetColumn(this, col, this.Rows);
                for (int i = 0; i < column.Length; i++)
                {
                    output[row, i] = column[i];
                }
                row++;
            }
            return output;
        }

        private static double[] GetColumn(Matrix matrix, int column, int rows)
        {
            var output = new double[rows];
            for (int row = 0; row < rows; row++)
            {
                output[row] = matrix[row, column];
            }
            return output;
        }

        public T[] Cast<T>()
        {
            var output = new T[this.Rows * this.Cols];
            int index = 0;
            for (int row = 0; row < this.Rows; row++)
            {
                for (int col = 0; col < this.Cols; col++)
                {
                    output[index] = (T)Convert.ChangeType(this[row, col], typeof(T));
                    index++;
                }
            }
            return output;
        }
    }
}
