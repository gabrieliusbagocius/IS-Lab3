using System;
using System.Diagnostics;

namespace Laboratory3
{
    public class Manager
    {
        private static Random r = new Random();

        public Random random
        {
            get { return r; }
            set { r = value; }
        }
        public static void Main()
        {
            var nn = new NeuralNetwork();

            double[,] trainingData = new double[20, 2];
            double inputs;
            double targets;

            for (var i = 0; i < trainingData.GetLength(0); i++)
            {
                trainingData[i, 0] = r.NextDouble();
                trainingData[i, 1] = (1 + 0.6 * Math.Sin(2 * Math.PI * trainingData[i, 0] / 0.7) + 0.3 * Math.Sin(2 * Math.PI * trainingData[i, 0])) / 2;
            }

            for (var n = 0; n < 1000; n++)
            {
                int rand = r.Next(0, trainingData.GetLength(0));
                inputs = trainingData[rand, 0];
                targets = trainingData[rand, 1];
                nn.Train(inputs, targets);
            }

            var inputVariables = new double[] { r.NextDouble(), r.NextDouble(), r.NextDouble(), r.NextDouble() };
            var received = nn.FeedForward(inputVariables[0]);
            Console.WriteLine("Received: " + received);
            Console.WriteLine("Actuals: " + (1 + (0.6 * Math.Sin(2 * Math.PI * inputVariables[0] / 0.7)) + (0.3 * Math.Sin(2 * Math.PI * inputVariables[0]))) / 2);
            Console.WriteLine();

            received = nn.FeedForward(inputVariables[1]);
            Console.WriteLine("Received: " + received);
            Console.WriteLine("Actuals: " + (1 + (0.6 * Math.Sin(2 * Math.PI * inputVariables[1] / 0.7)) + (0.3 * Math.Sin(2 * Math.PI * inputVariables[1]))) / 2);
            Console.WriteLine();

            received = nn.FeedForward(inputVariables[2]);
            Console.WriteLine("Received: " + received);
            Console.WriteLine("Actuals: " + (1 + (0.6 * Math.Sin(2 * Math.PI * inputVariables[2] / 0.7)) + (0.3 * Math.Sin(2 * Math.PI * inputVariables[2]))) / 2);
            Console.WriteLine();

            received = nn.FeedForward(inputVariables[3]);
            Console.WriteLine("Received: " + received);
            Console.WriteLine("Actuals: " + (1 + (0.6 * Math.Sin(2 * Math.PI * inputVariables[3] / 0.7)) + (0.3 * Math.Sin(2 * Math.PI * inputVariables[3]))) / 2);
            Console.WriteLine();

            Console.ReadKey();
        }

    }
}