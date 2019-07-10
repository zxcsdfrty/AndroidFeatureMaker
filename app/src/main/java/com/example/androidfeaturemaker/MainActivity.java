package com.example.androidfeaturemaker;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.icu.util.TimeZone;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.FpsMeter;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import static org.opencv.core.Core.add;
import static org.opencv.core.Core.log;
import static org.opencv.features2d.Features2d.drawKeypoints;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{

    JavaCameraView javaCameraView;
    private static String Tag = "MainActivity";

    FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.AKAZE);//opencv不支持SIFT、SURF检测方法
    DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.AKAZE);

    //match descriptor vectors
    DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
    MatOfDMatch matches = new MatOfDMatch();

    MatOfKeyPoint keyPoint_train =new MatOfKeyPoint();
    MatOfKeyPoint keyPoint_test =new MatOfKeyPoint();

    Mat descriptor1 =new Mat();
    Mat descriptor2 =new Mat();

    Mat mRgba = new Mat();
    Mat mGray = new Mat();

    Size size = new Size(200, 200); //size為200*200

    Mat oldpaste=new Mat();
    Mat paste = new Mat(size, CvType.CV_16S);
    Mat oldmaker= new Mat();
    Mat maker = new Mat(size, CvType.CV_16S);// 创建一个新的Mat（opencv的矩阵数据类型）
    Mat makerGray = new Mat();

    Mat obj_pixel = new Mat((int)size.height,(int)size.width,CvType.CV_32FC2);
    Mat scene_pixel = new Mat((int)size.height,(int)size.width,CvType.CV_32FC2);

    BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            super.onManagerConnected(status);
            switch (status) {
                case BaseLoaderCallback.SUCCESS: {
                    javaCameraView.enableView();
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    static {
        if (OpenCVLoader.initDebug()) {
            Log.i(Tag, "opencv loaded");
        } else {
            Log.i(Tag, "opencv not loaded");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Bitmap bmp = BitmapFactory.decodeResource(getResources(), R.drawable.p1);
        Bitmap bmp32 = bmp.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, oldmaker);
        Imgproc.resize(oldmaker, maker,size);//调用Imgproc的Resize方法，进行图片缩放

        bmp = BitmapFactory.decodeResource(getResources(), R.drawable.p2);
        bmp32 = bmp.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, oldpaste);
        Imgproc.resize(oldpaste, paste,size);// 將圖片大小設為跟maker一樣

        //將maker轉成灰階
        Imgproc.cvtColor(maker, makerGray, Imgproc.COLOR_RGB2GRAY);
        //偵測MAKER的keypoint and descriptor
        featureDetector.detect(makerGray,keyPoint_train);
        descriptorExtractor.compute(makerGray,keyPoint_train,descriptor1);

        //If authorisation not granted for camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            //ask for authorisation
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 50);
        //txt1 = (TextView) findViewById(R.id.txt1);
        javaCameraView = (JavaCameraView) findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (javaCameraView != null) {
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (javaCameraView != null) {
            javaCameraView.disableView();

        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            Log.i(Tag, "opencv loaded");
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);

        } else {
            Log.i(Tag, "opencv not loaded");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, baseLoaderCallback);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);//CV_(位元數)+(資料型態)+(Channel數)
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        //the camera view size & orientation issue can be fix in
        //CameraBridgeViewBase.java in opencv library
        //in the function "deliverAndDrawFrame"

        mRgba = inputFrame.rgba();
        mGray=inputFrame.gray();

        //偵測CAMERA的keypoint and descriptor*/
        featureDetector.detect(mGray,keyPoint_test);
        descriptorExtractor.compute(mGray,keyPoint_test,descriptor2);

        matcher.match(descriptor1, descriptor2, matches);
        List<DMatch> matchesList = matches.toList();

        //若沒有任何點match直接return,可以防止當matchesList沒有任何東西時outofboundary的情況
        if(matchesList.isEmpty())
            return mRgba;

        Double max_dist = 0.0;
        Double min_dist = 100.0;
        Log.i("descriptor","row: "+descriptor1.rows()+" col: "+descriptor1.cols());
        for(int i = 0; i < descriptor1.rows(); i++){
                Double dist = (double) matchesList.get(i).distance;
                if (dist < min_dist) min_dist = dist;
                if (dist > max_dist) max_dist = dist;
        }

        Log.i("distance","min: "+min_dist+" max: "+max_dist);

        LinkedList<DMatch> good_matches = new LinkedList<DMatch>();
        MatOfDMatch gm = new MatOfDMatch();
        int  count=0;
        //篩選特徵點
        for(int i = 0; i < descriptor1.rows(); i++){
            if(matchesList.get(i).distance < 3*min_dist){
                good_matches.addLast(matchesList.get(i));
                count++;
            }
        }
        Log.i("descriptorCount","count: " + count);

        if(count>50)//大於一定值代表沒偵測到maker test-----------------
            return mRgba;

        gm.fromList(good_matches);

        List<KeyPoint> keypoints_objectList = keyPoint_train.toList();
        List<KeyPoint> keypoints_sceneList = keyPoint_test.toList();

        LinkedList<Point> objList = new LinkedList<Point>();
        LinkedList<Point> sceneList = new LinkedList<Point>();
        //將匹配到的特徵點取出
        Log.i("good_matches.size","size: "+good_matches.size());
        for(int i = 0; i<good_matches.size(); i++){
            objList.addLast(keypoints_objectList.get(good_matches.get(i).queryIdx).pt);
            sceneList.addLast(keypoints_sceneList.get(good_matches.get(i).trainIdx).pt);
        }

        MatOfPoint2f obj = new MatOfPoint2f();
        obj.fromList(objList);

        MatOfPoint2f scene = new MatOfPoint2f();
        scene.fromList(sceneList);

        //找出實景跟maker的homography
        Mat hg = Calib3d.findHomography(obj, scene);

        /*
        Mat obj_corners = new Mat(4,1,CvType.CV_32FC2);
        Mat scene_corners = new Mat(4,1,CvType.CV_32FC2);

        obj_corners.put(0, 0, new double[] {0,0});
        obj_corners.put(1, 0, new double[] {makerGray.cols(),0});
        obj_corners.put(2, 0, new double[] {makerGray.cols(),makerGray.rows()});
        obj_corners.put(3, 0, new double[] {0,makerGray.rows()});

        //利用homography和maker已知的4個角來推出maker在實景中的位置
        Core.perspectiveTransform(obj_corners,scene_corners, hg);

        //劃出實景中maker邊線

        Imgproc.line(mRgba, new Point(scene_corners.get(0,0)), new Point(scene_corners.get(1,0)), new Scalar(0, 255, 255),4);
        Imgproc.line(mRgba, new Point(scene_corners.get(1,0)), new Point(scene_corners.get(2,0)), new Scalar(0, 255, 255),4);
        Imgproc.line(mRgba, new Point(scene_corners.get(2,0)), new Point(scene_corners.get(3,0)), new Scalar(0, 255, 255),4);
        Imgproc.line(mRgba, new Point(scene_corners.get(3,0)), new Point(scene_corners.get(0,0)), new Scalar(0, 255, 255),4);
        */

        //將obj與實景貼合
        for(int height=0;height<paste.rows();height++){
            for(int width=0;width<paste.cols();width++){
                //將貼圖(height,width)位置存下來
                obj_pixel.put(height, width, new double[] {height,width});
            }
        }
        //將貼圖(height,width)位置轉換成在實景的位置,存在scene_pixel中
        Core.perspectiveTransform(obj_pixel,scene_pixel, hg);
        //將obj的每一個pixel貼到實景上
        for(int height=0;height<paste.rows();height++){
            for(int width=0;width<paste.cols();width++){
                //將貼圖(height,width)位置的pixel的資料存下來
                double[] data=paste.get(height,width);
                Point point=new Point(scene_pixel.get(height,width));
                //Log.i("color","R:"+data[0]+" G:"+data[1]+" B:"+data[2]+" A:"+data[3]);
                //point.y對應row，point.x對應col,data[0-3] RGBA
                if(data[0]==255 && data[1]==255 && data[2]==255)//將obj白色部份去掉
                    continue;
                mRgba.put((int)point.y,(int)point.x,data[0],data[1],data[2],data[3]);
            }
        }
        return mRgba;
    }
}