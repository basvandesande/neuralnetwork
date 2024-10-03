using System.Text.Json.Serialization;
using System.Text.Json;

namespace NeuralNetwork
{
    public class TwoDimensionalDoubleArrayJsonConverter : JsonConverter<Matrix>
    {
        public override Matrix? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            using var jsonDoc = JsonDocument.ParseValue(ref reader);

            var rowLength = jsonDoc.RootElement.GetArrayLength();
            var columnLength = jsonDoc.RootElement.EnumerateArray().First().GetArrayLength();

            Matrix grid = new Matrix(rowLength, columnLength);

            int row = 0;
            foreach (var array in jsonDoc.RootElement.EnumerateArray())
            {
                int column = 0;
                foreach (var number in array.EnumerateArray())
                {
                    grid[row, column] = number.GetDouble();
                    column++;
                }
                row++;
            }

            return grid;
        }
        public override void Write(Utf8JsonWriter writer, Matrix value, JsonSerializerOptions options)
        {
            writer.WriteStartArray();
            for (int i = 0; i < value.Rows; i++)
            {
                writer.WriteStartArray();
                for (int j = 0; j < value.Cols; j++)
                {
                    writer.WriteNumberValue(value[i, j]);
                }
                writer.WriteEndArray();
            }
            writer.WriteEndArray();
        }
    }
}