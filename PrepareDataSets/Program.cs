using System.IO;

namespace PrepareDataSets
{
    public class Program
    {
        
        static void Main(string[] args)
        {
            // copy specific lines from an input file to an output file, e.g. lines 1-1000
            string inputFilePath = @"C:\Development\NeuralNetwork\mnist sets\mnist_train.csv";
            string outputFilePath = @"C:\Development\NeuralNetwork\mnist sets\mnist_train_.csv";
           
            CopyLines(inputFilePath, outputFilePath.Replace(".csv",  "100.csv"),     0,   101);
            CopyLines(inputFilePath, outputFilePath.Replace(".csv",  "200.csv"),   100,   300);
            CopyLines(inputFilePath, outputFilePath.Replace(".csv",  "500.csv"),   300,   800);
            CopyLines(inputFilePath, outputFilePath.Replace(".csv", "1000.csv"),  1000,  2000);
            CopyLines(inputFilePath, outputFilePath.Replace(".csv", "5000.csv"),  2000,  7000);
            CopyLines(inputFilePath, outputFilePath.Replace(".csv","10000.csv"), 10000, 20000);

            Console.WriteLine("Done.");
        }
        // create a copylines function for me with the following signature CopyLines(inputFilePath, outputFilePath, startLine, endLine)
        public static void CopyLines(string inputFilePath, string outputFilePath, int startLine, int endLine)
        {
            Console.WriteLine($"Copying lines {startLine} to {endLine} from {inputFilePath} to {outputFilePath}.");	
            using (StreamReader reader = new StreamReader(inputFilePath))
            {
                using (StreamWriter writer = new StreamWriter(outputFilePath))
                {
                    string line;
                    int lineNumber = 0;
                    while ((line = reader.ReadLine()) != null)
                    {
                        lineNumber++;
                        if (lineNumber >= startLine && lineNumber < endLine)
                        {
                            writer.WriteLine(line);
                        }
                    }
                }
            }
        }


    }
}