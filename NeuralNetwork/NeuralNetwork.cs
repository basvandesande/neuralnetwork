using Newtonsoft.Json;

namespace NeuralNetwork
{
    internal class NeuralNetwork
    {
        private int InputNodes { get; set; } = 3;
        private int HiddenNodes { get; set; } = 3;
        private int OutputNodes { get; set; } = 3;
        private double LearningRate { get; set; } = 0.5;
        
        private Matrix _weightInputToHidden;
        private Matrix _weightHiddenToOutput;

        private int _totalTrainingRecords = 0;
        private int _totalTestRecords = 0;
        private int _correct = 0;

        public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate)
        {
            InputNodes = inputNodes;    
            HiddenNodes = hiddenNodes;
            OutputNodes = outputNodes;
            LearningRate = learningRate;

            _weightInputToHidden = InitializeBiasedMatrix(HiddenNodes, InputNodes);
            _weightHiddenToOutput = InitializeBiasedMatrix(OutputNodes, HiddenNodes);
        }
        
        private Matrix InitializeBiasedMatrix(int rows, int cols)
        {
            Random random = new(Environment.TickCount);
            var output = new Matrix(rows, cols);

            for (int row=0; row<rows; row++)
            {
                for (int col = 0; col < cols; col++)
                {
                    // we want to have positive and negative values between -0.5 and 0.5
                    output[row, col] = random.NextDouble() - 0.5;
                }
            }
            return output;
        }

        /// <summary>
        /// The Sigmoid() is the activation function
        /// </summary>
        private double Sigmoid(double x) =>  1.0 / (1.0 + Math.Exp(-x));
        
        public double[] Query(double[] input_list)
        {
            // feed the neural network with the input data (convert 1d array to 2d matrix and transpose)
            Matrix inputs = new Matrix(input_list).Transpose();

            // process the input and pass it through the hidden layer, sigmoid (activation function) will fire if needed
            Matrix hidden_inputs = _weightInputToHidden * inputs;
            Matrix hidden_outputs = hidden_inputs.ApplyActivator(Sigmoid);

            // process the output of the hidden layer and pass it through the the output layer, sigmoid (activation function) will fire if needed
            Matrix final_inputs = _weightHiddenToOutput * hidden_outputs;
            Matrix final_outputs = final_inputs.ApplyActivator(Sigmoid);
            
            // flatten the outputs to a 1D array and return it
            return final_outputs.Cast<double>().ToArray(); 
        }

        public (int targetLabel, int label) Test(double[] input_list, double[] targets_list)
        {
            double[] outputs = Query(input_list);
           
            // get the indexes of the max value from the final_outputs and the perfect score from the targets_list (0.99)
            int indexMaxIndexOutputs = outputs.ToList().IndexOf(outputs.Max());
            int indexMaxTargetLabel = targets_list.ToList().IndexOf(targets_list.Max());

            // determine statistics for the test
            _totalTestRecords++;
            _correct += (indexMaxTargetLabel == indexMaxIndexOutputs) ? 1 : 0;

            // return target label and the label that was found
            return (indexMaxTargetLabel, indexMaxIndexOutputs); 
        }

        public void Train(double[] input_list, double[] targets_list)
        {
            // train the network with the input data
            Matrix inputs = new Matrix(input_list).Transpose();

            Matrix hidden_inputs = _weightInputToHidden * inputs;
            Matrix hidden_outputs = hidden_inputs.ApplyActivator(Sigmoid);

            Matrix final_inputs = _weightHiddenToOutput * hidden_outputs;
            Matrix final_outputs = final_inputs.ApplyActivator(Sigmoid);

            // the output layer error is the (target - actual outcome (final_outputs))
            // update the weights for the links between the hidden and the output layers
            Matrix targets = new Matrix(targets_list).Transpose();
            Matrix output_errors = targets - final_outputs;
            UpdateWeights(ref _weightHiddenToOutput, output_errors, final_outputs, hidden_outputs);

            // hidden layer error is the output_errors, split by weights, recombined at the hidden_nodes
            // update the weights for the links between the input and the hidden layers
            Matrix hidden_errors = _weightHiddenToOutput.Transpose() * output_errors;
            UpdateWeights(ref _weightInputToHidden,  hidden_errors, hidden_outputs, inputs);

            // set statistics for the training
            _totalTrainingRecords++;
        }

        private void UpdateWeights(ref Matrix weights, Matrix errors, Matrix outputs, Matrix nodeLayer)
        {
            // We receive 2 matrixes with the same shape for errors and outputs. We treat them as 1 dimensional arrays (1 element per row)
            // Calculate new "outputs" based on the learning rate, outputs and the errors  => learningrate* errors * outputs * (1.0 - outputs)
            for (int row = 0; row < outputs.Rows; row++)
            {
                outputs[row, 0] = LearningRate * errors[row, 0] * outputs[row, 0] * (1.0 - outputs[row, 0]);
            }

            // Multiply the new "outputs" (which is error corrected) matrix with the transposed nodeLayer matrix and put it in backpropagated matrix
            Matrix backpropagated = outputs * nodeLayer.Transpose();

            // Add the backpropagrated matrix to the weights matrix. At this point we enforce and weaken neural connections (it's learning!) 
            weights += backpropagated;
        }

        public void Transform(ref double[] input, bool isTrainingData)
        {
            // if the line is used for training then we skip the first number in the line
            // per number that is read, divide it by 255.0 then multiply it by 0.99 and add 0.01
            // this way we have a range of values between 0.01 and 1.00. (input with a value of 0 is now detectable) 
            int offSet = isTrainingData ? 1 : 0;

            for (int i = offSet; i < input.Length; i++)
            {
                input[i] = (Convert.ToDouble(input[i]) / 255.0 * 0.99) + 0.01;
            }
            if (isTrainingData)
            {
                input = input[1..];   // remove the first element from the array
            }
        }

        public double[] GetTargetsList(int desired_value)
        {
            // for each output node, set the target value to 0.01 (the lowest possible value)
            // and set the target value for the node that represents the value to 0.99
            var outputs = new double[OutputNodes];
            for (int i = 0; i < OutputNodes; i++)
            {
                outputs[i] = 0.01;
            }
            // 0 based index, enumerations from 0 to n.... (label has no semantic meaning)
            outputs[desired_value] = 0.99;

            return outputs;
        }

        public void FromJson(string json)
        {
            var data = JsonConvert.DeserializeObject<NeuralNetworkData>(json) ?? throw new ArgumentException("Invalid JSON data", nameof(json));

            InputNodes = data.InputNodes;
            HiddenNodes = data.HiddenNodes;
            OutputNodes = data.OutputNodes;
            LearningRate = data.LearningRate;

            _weightInputToHidden = data.WeightInputToHidden;
            _weightHiddenToOutput = data.WeightHiddenToOutput;
        }

        public string ToJson()
        {
            var data = new NeuralNetworkData
            {
                InputNodes = InputNodes,
                HiddenNodes = HiddenNodes,
                OutputNodes = OutputNodes,
                LearningRate = LearningRate,
                Statistics = new NeuralNetworkStatistics
                {
                    TrainingRecords = _totalTrainingRecords,
                    TestRecords = _totalTestRecords,
                    Correct = _correct
                },
                WeightInputToHidden = _weightInputToHidden,
                WeightHiddenToOutput = _weightHiddenToOutput
            };
            return JsonConvert.SerializeObject(data);
        }
    }
}
