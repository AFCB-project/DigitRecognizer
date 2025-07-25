using System;
using System.IO;
using DigitRecognizer.AI;

namespace DigitRecognizer.AI
{
    class Program
    {
        static NeuralNetwork net;
        static string root = AppDomain.CurrentDomain.BaseDirectory;
        static string dataFolder = Path.Combine(root, "..", "..", "Datasource");
        static string inputImagePath = Path.Combine(root, "..", "..", "Data", "Pic.png");

        static void Main(string[] args)
        {
            Console.WriteLine("=== DigitRecognizer v1.0 ===");
            Console.WriteLine("Neural network starts from scratch.\n");

            net = new NeuralNetwork(inputSize: 784, hiddenSize: 128, outputSize: 10);

            LoadTrainingData();

            while (true)
            {
                if (!File.Exists(inputImagePath))
                {
                    Console.WriteLine("❗ Place Pic.png inside the 'Data' folder and press Enter...");
                    Console.ReadLine();
                    continue;
                }

                float[] input = ImageProcessor.LoadImageAsInput(inputImagePath);
                int prediction = net.Predict(input);

                Console.WriteLine($"\n🤖 I think this is: {prediction}");
                Console.Write("Was I correct? (1 = yes, 0 = no): ");
                string response = Console.ReadLine();

                if (response == "1")
                {
                    Console.WriteLine("✅ Great!");
                }
                else
                {
                    Console.Write("What was the correct digit (0–9)? ");
                    string correctStr = Console.ReadLine();

                    if (int.TryParse(correctStr, out int correctLabel) && correctLabel >= 0 && correctLabel <= 9)
                    {
                        SaveNewSample(correctLabel);
                        net.Train(input, correctLabel, learningRate: 0.1f);
                        Console.WriteLine("🔁 Learned from this mistake!");
                    }
                    else
                    {
                        Console.WriteLine("❌ Invalid label.");
                    }
                }

                Console.WriteLine("\n📎 Press Enter to continue with a new image...");
                Console.ReadLine();
            }
        }

        static void LoadTrainingData()
        {
            if (!Directory.Exists(dataFolder))
            {
                Directory.CreateDirectory(dataFolder);
                Console.WriteLine("🆕 Datasource folder created.");
                return;
            }

            string[] files = Directory.GetFiles(dataFolder, "Data*_*.png");
            int trained = 0;

            foreach (var file in files)
            {
                string name = Path.GetFileNameWithoutExtension(file); // e.g., Data7_3
                string[] parts = name.Split('_');
                if (parts.Length < 2) continue;

                if (int.TryParse(parts[1], out int label))
                {
                    float[] input = ImageProcessor.LoadImageAsInput(file);
                    net.Train(input, label, learningRate: 0.1f);
                    trained++;
                }
            }

            Console.WriteLine($"📚 Loaded and trained on {trained} saved samples.");
        }

        static void SaveNewSample(int label)
        {
            if (!Directory.Exists(dataFolder))
                Directory.CreateDirectory(dataFolder);

            int index = 1;
            string fileName;

            do
            {
                fileName = Path.Combine(dataFolder, $"Data{index}_{label}.png");
                index++;
            } while (File.Exists(fileName));

            File.Copy(inputImagePath, fileName);
            Console.WriteLine($"💾 Saved to: {Path.GetFileName(fileName)}");
        }
    }
}
