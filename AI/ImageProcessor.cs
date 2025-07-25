using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;

namespace DigitRecognizer.AI
{
    public static class ImageProcessor
    {
        public static float[] LoadImageAsInput(string path, int width = 28, int height = 28)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"Nohing found: {path}");

            using (Bitmap original = new Bitmap(path))
            using (Bitmap resized = new Bitmap(width, height))
            {
                using (Graphics g = Graphics.FromImage(resized))
                {
                    g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    g.DrawImage(original, 0, 0, width, height);
                }

                float[] input = new float[width * height];

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        Color pixel = resized.GetPixel(x, y);

                        float gray = (pixel.R + pixel.G + pixel.B) / 3f;

                        gray = 255f - gray;

                        float normalized = gray / 255f;

                        input[y * width + x] = normalized;
                    }
                }

                return input;
            }
        }
    }
}
