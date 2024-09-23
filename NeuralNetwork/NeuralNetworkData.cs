namespace NeuralNetwork
{
    public record NeuralNetworkData
    {
        public int InputNodes { get; init; }
        public int HiddenNodes { get; init; }
        public int OutputNodes { get; init; }
        public double LearningRate { get; init; }
        public required NeuralNetworkStatistics Statistics { get; set; }
        public required Matrix WeightInputToHidden { get; init; }
        public required Matrix WeightHiddenToOutput { get; init; }
    }
    public record NeuralNetworkStatistics
    {
        public int TrainingRecords { get; set; } = 0;
        public int TestRecords { get; set; } = 0;
        public int Correct { get; set; } = 0;
        public decimal Accuracy => (decimal)Correct / (TestRecords == 0 ? 1 : TestRecords);
    }

}
