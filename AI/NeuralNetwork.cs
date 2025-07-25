using System;

namespace DigitRecognizer.AI
{
    public class NeuralNetwork
    {
        private int inputSize, hiddenSize, outputSize;

        private float[] inputLayer;
        private float[] hiddenLayer;
        private float[] outputLayer;

        private float[,] weightsInputHidden;
        private float[,] weightsHiddenOutput;

        private float[] biasHidden;
        private float[] biasOutput;

        private Random rand;

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;

            inputLayer = new float[inputSize];
            hiddenLayer = new float[hiddenSize];
            outputLayer = new float[outputSize];

            weightsInputHidden = new float[inputSize, hiddenSize];
            weightsHiddenOutput = new float[hiddenSize, outputSize];

            biasHidden = new float[hiddenSize];
            biasOutput = new float[outputSize];

            rand = new Random();
            InitializeWeights();
        }

        private void InitializeWeights()
        {
            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < hiddenSize; j++)
                    weightsInputHidden[i, j] = (float)(rand.NextDouble() - 0.5);

            for (int i = 0; i < hiddenSize; i++)
                for (int j = 0; j < outputSize; j++)
                    weightsHiddenOutput[i, j] = (float)(rand.NextDouble() - 0.5);

            for (int i = 0; i < hiddenSize; i++)
                biasHidden[i] = 0;

            for (int i = 0; i < outputSize; i++)
                biasOutput[i] = 0;
        }

        public int Predict(float[] input)
        {
            float[] output = Feedforward(input);
            int predicted = 0;
            float max = output[0];
            for (int i = 1; i < output.Length; i++)
            {
                if (output[i] > max)
                {
                    max = output[i];
                    predicted = i;
                }
            }
            return predicted;
        }

        public void Train(float[] input, int correctLabel, float learningRate)
        {
            Feedforward(input);

            float[] target = new float[outputSize];
            target[correctLabel] = 1f;

            float[] outputErrors = new float[outputSize];
            for (int i = 0; i < outputSize; i++)
                outputErrors[i] = outputLayer[i] - target[i];

            for (int i = 0; i < hiddenSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    float gradient = outputErrors[j] * outputLayer[j] * (1 - outputLayer[j]);
                    weightsHiddenOutput[i, j] -= learningRate * gradient * hiddenLayer[i];
                }
            }

            for (int i = 0; i < outputSize; i++)
            {
                float gradient = outputErrors[i] * outputLayer[i] * (1 - outputLayer[i]);
                biasOutput[i] -= learningRate * gradient;
            }

            float[] hiddenErrors = new float[hiddenSize];
            for (int i = 0; i < hiddenSize; i++)
            {
                float error = 0f;
                for (int j = 0; j < outputSize; j++)
                    error += outputErrors[j] * weightsHiddenOutput[i, j];
                hiddenErrors[i] = error;
            }

            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    float gradient = hiddenErrors[j] * ReLUDerivative(hiddenLayer[j]);
                    weightsInputHidden[i, j] -= learningRate * gradient * input[i];
                }
            }

            for (int i = 0; i < hiddenSize; i++)
            {
                float gradient = hiddenErrors[i] * ReLUDerivative(hiddenLayer[i]);
                biasHidden[i] -= learningRate * gradient;
            }
        }

        private float[] Feedforward(float[] input)
        {
            inputLayer = input;

            for (int i = 0; i < hiddenSize; i++)
            {
                float sum = 0;
                for (int j = 0; j < inputSize; j++)
                    sum += input[j] * weightsInputHidden[j, i];
                sum += biasHidden[i];
                hiddenLayer[i] = ReLU(sum);
            }

            for (int i = 0; i < outputSize; i++)
            {
                float sum = 0;
                for (int j = 0; j < hiddenSize; j++)
                    sum += hiddenLayer[j] * weightsHiddenOutput[j, i];
                sum += biasOutput[i];
                outputLayer[i] = sum;
            }

            Softmax(outputLayer);
            return outputLayer;
        }

        private float ReLU(float x) => Math.Max(0, x);
        private float ReLUDerivative(float x) => x > 0 ? 1 : 0;

        private void Softmax(float[] values)
        {
            float max = float.NegativeInfinity;
            foreach (float v in values)
                if (v > max) max = v;

            float sumExp = 0f;
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (float)Math.Exp(values[i] - max);
                sumExp += values[i];
            }

            for (int i = 0; i < values.Length; i++)
                values[i] /= sumExp;
        }
    }
}
