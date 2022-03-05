/*
    Teste de metodo proposto no artigo https://www.researchgate.net/publication/311402007_Skin_Color_Segmentation_Using_Multi-Color_Space_Threshold/link/5e46c493458515072d9da7db/download
    para a detecao de cor da pele e remocao da cor do fundo
 */
using System.Drawing;

namespace SEL
{
    public class Program
    {
        static void Main(string[] args)
        {

            using (Bitmap image = new Bitmap("C:/Users/artem/Desktop/SE/hand2.jpg"))
            {
                Bitmap iCopy = NormalizeRG(image);
                iCopy.Save("C:/Users/artem/Desktop/SE/nImage.jpg"); //Normalized Image

                //Calculo de HSV


            }
        }

        public static Bitmap NormalizeRG(Bitmap image)
        {
            Bitmap iCopy = new Bitmap(image.Width, image.Height);
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    Color pixel = image.GetPixel(x, y);
                    int nR = pixel.R / (pixel.R + pixel.G + pixel.B);
                    int nG = pixel.G / (pixel.R + pixel.G + pixel.B);
                    int nB = pixel.G / (pixel.R + pixel.G + pixel.B);

                    iCopy.SetPixel(x, y, Color.FromArgb(nR, nG, pixel.B));
                }
            }
            return iCopy;
        }

        public static void SaveMyImage(Bitmap image)
        {
            image.Save("C:/Users/artem/Desktop/SE/test.jpg");
        }
    }
}