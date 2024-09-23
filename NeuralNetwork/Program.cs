using System.Diagnostics;

namespace NeuralNetwork
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Neural network - object recognition");

            var stopWatch = new Stopwatch();
            var cls = new NeuralNetwork(784, 100, 10, 0.25);
            
            var emptyStateFilePath = @$"C:\Development\NeuralNetwork\NetworkState\learning rate 0_1\neural_state_100.json";

            //File.WriteAllText(emptyStateFilePath, cls.ToJson());
            string json = File.ReadAllText(emptyStateFilePath);
            cls.FromJson(json);
            
            string setSize = "60000";
            int maxEpochs = 20;
            int bestScore = 0;
            int epochs = 0;
            stopWatch.Start();
            
            for (epochs = 0; epochs < maxEpochs; epochs++)
            {
                Console.WriteLine($"- Training network (set {setSize}) - epoch {epochs + 1}...");
                TrainFromFile(cls, @$"C:\Development\NeuralNetwork\mnist sets\mnist_train_{setSize}.csv");

                Console.WriteLine("  Testing network (set 10000)...");
                int score = TestFromFile(cls, @"C:\Development\NeuralNetwork\mnist sets\mnist_test.csv", false);
                // bail out, in case we don't improve anymore
                if (score <= bestScore)
                {
                    Console.WriteLine("\r\n[Aborted, score did not improve]\r\n");
                    break;
                }
                bestScore = score;
            }


            // Save the state of the neural network
            var stateFilePath = @$"C:\development\neuralnetwork\networkstate\neural_state_{setSize}.json";
            File.WriteAllText(stateFilePath, cls.ToJson());

            stopWatch.Stop();
            Console.WriteLine($"- Training completed in {stopWatch.ElapsedMilliseconds / 1000} seconds for {setSize} objects in {epochs + 1} epochs\r\n");

            Console.WriteLine("\r\nNetwork trained and tested");
            Console.ReadLine();
        }

        private static void TrainFromFile(NeuralNetwork cls, string filePath)
        {
            using (StreamReader sr = new(filePath))
            {
                string? line;
                while ((line = sr.ReadLine()) != null)
                {
                    string[] values = line.Split(',');
                    double[] trainData = Array.ConvertAll(values, double.Parse);
                    double[] targets = cls.GetTargetsList((int)trainData[0]);
                    cls.Transform(ref trainData, true);
                    cls.Train(trainData, targets);
                }
            }
        }

        private static int TestFromFile(NeuralNetwork cls, string filePath, bool show=true)
        {
            List<(int targetLabel, int testLabel)> testResults = [];

            using (StreamReader sr = new(filePath))
            {
                string? line;
                while ((line = sr.ReadLine()) != null)
                {
                    string[] values = line.Split(',');
                    double[] testData = Array.ConvertAll(values, double.Parse);
                    int label = (int)testData[0];
                    double[] targets = cls.GetTargetsList((int)testData[0]);
                    cls.Transform(ref testData, true);

                    // build a list of test results
                    testResults.Add(cls.Test(testData, targets));
                }
            }

            // print the test results
            int score = 0;
            foreach (var (label, test) in testResults)
            {
                score += (label == test) ? 1 : 0;
                if (show)
                {
                    string result = (label == test) ? "[SUCCESS]" : "[FAILURE]";
                    Console.ForegroundColor = (label == test) ? ConsoleColor.Green : ConsoleColor.Red;
                    Console.Write($"  {result}  ");
                    Console.ResetColor();
                    Console.WriteLine($" Label: {label} - Test: {test}");
                }
            }

            decimal percentage = (decimal)score / (decimal)testResults.Count;
            Console.WriteLine($"  Score: {percentage:P2}  {score}/{testResults.Count}\r\n");

            return score;
        }



    }
}
