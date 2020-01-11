using System;
using System.Diagnostics;

namespace Laboratory3
{
    public class NeuralNetwork
    {
        private double[] weight = new double[3];
        private int bias;
        private double learningRate;
        private double r1;
        private double r2;
        private double c1;
        private double c2;

        public NeuralNetwork()
        {
            Manager manager = new Manager();
            Random r = manager.random;
            weight[0] = r.NextDouble();
            weight[1] = r.NextDouble();
            weight[2] = r.NextDouble();
            r1 = r.NextDouble();
            r2 = r.NextDouble();
            c1 = r.NextDouble();
            c2= r.NextDouble();
            bias = 1;
            learningRate = 0.1;
        }

        public double Train(double input, double output)
        {
            double predicted = CalculateOutput(input, weight, r1, r2, c1, c2);
            double localError = Math.Abs(output - predicted);

            if (Math.Abs(localError) > 0.01)
            {
                if (localError > 1) localError *= -1;
                weight[1] = Math.Abs(weight[1] + learningRate * localError * input);
                weight[2] = Math.Abs(weight[2] + learningRate * localError * input);
                weight[0] = Math.Abs(weight[0] + learningRate * localError * bias);
            }
            return localError;

        }
        public double FeedForward(double input)
        {
            return CalculateOutput(input, weight, r1, r2, c1, c2);
        }

        private static double CalculateOutput(double input, double[] weight, double r1, double r2, double c1, double c2)
        {
            double sumFirstInput = Math.Exp(((input - c1) * (input - c1)) / (2 * r1 * r1)) * weight[1];
            double sumSecondInput = Math.Exp(((input - c2) * (input - c2)) / (2 * r2 * r2)) * weight[2];
            double sumAdditional = 1 * weight[0];
            double sum = sumFirstInput + sumSecondInput + sumAdditional;
            return sum;
        }

    }
}