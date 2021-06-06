package com.example.opencvandroid;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.FileUtils;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    File mCascadeFile;
    CascadeClassifier cascadeClassifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        loadHaarCascaseFile();
        OpenCVLoader.initDebug();
    }

    public void displayToast(View v) {
        Mat img = null;

        try {
            img = Utils.loadResource(getApplicationContext(), R.drawable.headshot);
        } catch (IOException e) {
            e.printStackTrace();
        }

        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2BGRA);

        Mat img_result = img.clone();
        Imgproc.Canny(img, img_result, 80, 90);
        Bitmap img_bitmap = Bitmap.createBitmap(img_result.cols(), img_result.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img_result, img_bitmap);
        ImageView imageView = findViewById(R.id.image);
        imageView.setImageBitmap(img_bitmap);
    }

    public void detectFace(View v) {
        Mat img = null;

        try {
            img = Utils.loadResource(getApplicationContext(), R.drawable.headshot);
        } catch (IOException e) {
            e.printStackTrace();
        }

        cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        cascadeClassifier.load(mCascadeFile.getAbsolutePath());

        Mat gray = new Mat();

        Imgproc.cvtColor(img, gray, Imgproc.COLOR_RGB2GRAY);

        MatOfRect faces = new MatOfRect();

        cascadeClassifier.detectMultiScale(gray, faces);



        for (Rect rect : faces.toArray()) {
            Imgproc.rectangle(img, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0,0,255), 3);
        }

        Bitmap img_bitmap = Bitmap.createBitmap(img.cols(), img.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img, img_bitmap);
        ImageView imageView = findViewById(R.id.image);
        imageView.setImageBitmap(img_bitmap);

    }

    private void loadHaarCascaseFile() {
        try {
            File cascadeDir = getDir("haarcascade_frontalface_default.xml", Context.MODE_PRIVATE);
            mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");

            if(!mCascadeFile.exists()) {
                FileOutputStream os = new FileOutputStream(mCascadeFile);
                InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_default);
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }
                is.close();
                os.close();
            }
        } catch (Throwable throwable) {
            throw new RuntimeException("Failed to load Haar Cascade file");
        }
    }
}