﻿/*
    O artigo que estava a referir nao explica bem o algoritmo e faltam passos pelo que nao esta implementado como deve ser
    Preferi usar novo artigo https://www.ripublication.com/ijaer17/ijaerv12n6_35.pdf

 */
using System.Drawing;
using ColorHelper;

namespace SEL
{
    public class Program
    {
        public class CGCR
        {
            public float Y { get; set; }    
            public float Cg { get;set; }
            public float Cr { get; set; }
        }


        static void Main(string[] args)
        {

            using (Bitmap image = new Bitmap("C:/Users/artem/Desktop/SE/hand3.jpg"))
            {
                Bitmap iCopy = NormalizeRG(image);

                float avgR=0, avgG=0, avgB=0;
                float aR, aG, aB, avgGray;

                for (int y = 0; y < image.Height; y++)
                {
                    for (int x = 0; x < image.Width; x++)
                    {
                        avgR+=image.GetPixel(x, y).R;
                        avgG+=image.GetPixel(x, y).G;
                        avgB+=image.GetPixel(x, y).B;
                    }
                }
                int rounds = image.Height * image.Width;
                avgR = avgR / rounds;
                avgG = avgG /rounds;
                avgB = avgB / rounds;

                avgGray = (avgR + avgG + avgB) / 3;

                aR = avgGray/ avgR;
                aG = avgGray/ avgG;
                aB = avgGray/ avgB;
                
                for (int y = 0; y < image.Height; y++)
                {
                    for (int x = 0; x < image.Width; x++)
                    {
                        float R, G, B;

                        R = (image.GetPixel(x, y).R * aR < 255) ? (image.GetPixel(x, y).R * aR) : 255;  
                        G = (image.GetPixel(x, y).G * aG < 255) ? (image.GetPixel(x, y).G * aG) : 255;  
                        B = (image.GetPixel(x, y).B * aB < 255) ? (image.GetPixel(x, y).B * aB) : 255;  

                        //Console.WriteLine($"{R} {G} {B}");
                        image.SetPixel(x, y, Color.FromArgb((int)R, (int)G, (int)B));
                    }
                }


                for (int y = 0; y < image.Height; y++)
                {
                    for (int x = 0; x < image.Width; x++)
                    {
                        if (RgbToYCgCr(image.GetPixel(x,y)))
                        {
                            image.SetPixel(x,y, Color.FromArgb(255,255,255));
                            
                            continue;
                        }
                        image.SetPixel(x, y, Color.FromArgb(0, 0, 0));
                    }
                }


                image.Save("C:/Users/artem/Desktop/SE/nImage.jpg"); //Normalized Image

                //Calculo de HSV


            }
        }
        //Segundo o artigo usamos um threshhold fixo
        public static bool RgbToYCgCr(Color pixel)
        {
            CGCR newPixel = new CGCR();
            float value;
            newPixel.Y = (float)(16 + 65.481 * pixel.R/256 + 128.553 * pixel.G / 256 + 24.966 * pixel.B / 256);
            newPixel.Cg = (float)(128 + -81.085 * pixel.R / 256 + 112 * pixel.G / 256 + -30.915 * pixel.B / 256);
            newPixel.Cr = (float)(128 + 112 * pixel.R / 256 + -93.786 * pixel.G / 256 + -18.214 * pixel.B / 256);

            /*
             {Cg∈[85, 135] && Cr∈[−Cg + 260, −Cg + 280]}
             */
            
            return ((85 <= newPixel.Cg && newPixel.Cg <= 135) && (((-newPixel.Cg + 260) <= newPixel.Cr) && (newPixel.Cr <= (-newPixel.Cg + 280))));
        }

        public static void SaveMyImage(Bitmap image)
        {
            image.Save("C:/Users/artem/Desktop/SE/test.jpg");
        }



        //Lixo
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
    }
}