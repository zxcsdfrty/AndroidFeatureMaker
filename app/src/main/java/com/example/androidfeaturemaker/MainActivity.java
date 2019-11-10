package com.example.androidfeaturemaker;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ColorSpace;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.icu.util.TimeZone;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
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
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.Reader;
import java.net.Socket;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.lang.Thread;
import java.text.DecimalFormat;

import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.TRUE;
import static org.opencv.core.Core.add;
import static org.opencv.core.Core.bitwise_and;
import static org.opencv.core.Core.eigen;
import static org.opencv.core.Core.gemm;
import static org.opencv.core.Core.log;
import static org.opencv.core.Core.pow;
import static org.opencv.core.Core.sqrt;
import static org.opencv.core.Core.trace;
import static org.opencv.core.CvType.CV_16SC3;
import static org.opencv.core.CvType.CV_16UC3;
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.core.CvType.CV_8UC4;
import static org.opencv.features2d.DescriptorMatcher.BRUTEFORCE_HAMMING;
import static org.opencv.features2d.Features2d.drawKeypoints;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2BGR;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2BGRA;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2GRAY;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2RGBA;
import static org.opencv.imgproc.Imgproc.COLOR_RGBA2BGR;
import static org.opencv.imgproc.Imgproc.COLOR_RGBA2BGRA;
import static org.opencv.imgproc.Imgproc.COLOR_RGBA2RGB;
import static org.opencv.imgproc.Imgproc.CV_RGBA2mRGBA;
import static org.opencv.imgproc.Imgproc.circle;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    protected static final float FLIP_DISTANCE = 150;
    private ExecutorService mThreadPool;
    //網路串流
    InputStream inputStream;
    Thread connectServer;
    Thread transmission;
    private Socket socket;
    OutputStream outputStream;
    boolean checkConnect = false;
    boolean beginG = false;
    int datasize;
    //byte[][] data = new byte[2][];
    byte[] data;
    //紀錄登入序號
    int playerList = -1;
    //紀錄玩家資訊
    boolean ready = false;
    int move = -1;
    String st;
    String st1 = "45220";
    //test
    private Button btnSend;
    private Button btnReady;
    private Button btnClose;
    private TextView txvcolor;
    private TextView txvspeed;
    private TextView txvsize;
    private TextView txvalive;
    private TextView txvresult;
    ImageView imgView;
    private Handler mMainHandler;
    Bitmap bmp;
    Bitmap bmp32;
    //速度優化
    DecimalFormat decimalFormat = new DecimalFormat("##.000");
    Lock lock = new ReentrantLock();

    //解析度在JavaCameraView調
    JavaCameraView javaCameraView;
    private static String Tag = "MainActivity";
    GestureDetector mGesture;
    FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.AKAZE);//opencv不支持SIFT、SURF检测方法
    DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.AKAZE);
    //match descriptor vectors
    DescriptorMatcher matcher = DescriptorMatcher.create(BRUTEFORCE_HAMMING);
    MatOfDMatch matches = new MatOfDMatch();
    MatOfKeyPoint keyPoint_train = new MatOfKeyPoint();
    MatOfKeyPoint keyPoint_test = new MatOfKeyPoint();
    Mat descriptor1 = new Mat();
    Mat descriptor2 = new Mat();
    Mat mRgba = new Mat();
    Mat mGray = new Mat();
    Mat paste = new Mat();
    Mat pasteGray = new Mat();
    Mat maker = new Mat();
    Mat cameraMatrix = new Mat(3, 3, CvType.CV_32F);//CV_32F：32-bit ﬂoating-point numbers
    MatOfDouble distCoeffs = new MatOfDouble();
    Mat Tvec = new Mat();
    Mat Rvec = new Mat();
    Point3 Position = new Point3();
    Point3 UnityPosition = new Point3();

    MatOfByte statusMat = new MatOfByte();
    MatOfFloat errSimilarityMat = new MatOfFloat();
    MatOfPoint2f estimateScenePoint = new MatOfPoint2f();
    MatOfPoint3f estimateMakerPoints = new MatOfPoint3f();
    Mat estimateFrame = new Mat();
    //lightFrame完取完特徵點當前的estimateFrame
    Mat lightFrame = new Mat();
    //當前畫面的上1禎
    Mat frameBuffer = new Mat();
    //上1張的paste
    Mat pasteBuffer = new Mat();
    //是否偵測到maker
    Boolean DETECTTOMAKER = FALSE;


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
        btnSend = (Button) findViewById(R.id.Begin);
        btnReady = (Button) findViewById(R.id.Ready);
        btnClose = (Button) findViewById(R.id.close);
        imgView = (ImageView)findViewById(R.id.image);
        txvcolor = (TextView) findViewById(R.id.color);
        txvspeed = (TextView) findViewById(R.id.speed);
        txvsize = (TextView) findViewById(R.id.size);
        txvalive = (TextView) findViewById(R.id.IsAlive);
        txvresult = (TextView) findViewById(R.id.IsWin);
        /*Button calibration = (Button) findViewById(R.id.Calibration);
        calibration.setOnClickListener(new Button.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setClass(MainActivity.this, Calibration.class);
                startActivity(intent);
            }
        });*/
        bmp = BitmapFactory.decodeResource(getResources(), R.drawable.p1);
        bmp32 = bmp.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, maker);
        Imgproc.resize(maker, maker, new Size(maker.cols() / 8, maker.rows() / 8));//调用Imgproc的Resize方法，进行图片缩放
        //Imgproc.pyrDown(maker, maker, new Size(maker.cols()/2, maker.rows()/2));
        Log.i("imageSize", maker.rows() + " " + maker.cols());
        //將maker轉成灰階
        Imgproc.cvtColor(maker, maker, COLOR_RGB2GRAY);
        //偵測MAKER的keypoint and descriptor
        featureDetector.detect(maker, keyPoint_train);
        descriptorExtractor.compute(maker, keyPoint_train, descriptor1);
        //If authorisation not granted for camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            //ask for authorisation
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 50);
        javaCameraView = (JavaCameraView) findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);

        Thread positionEstimate = new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    //frameBuffer是當前畫面的上一禎
                    if (frameBuffer.empty())
                        continue;
                    //相機內部參數
                    cameraMatrix.put(0, 0, 550 * mRgba.cols() / 720);
                    cameraMatrix.put(0, 1, 0);
                    cameraMatrix.put(0, 2, mRgba.cols() / 2);
                    cameraMatrix.put(1, 0, 0);
                    cameraMatrix.put(1, 1, 550 * mRgba.cols() / 720);
                    cameraMatrix.put(1, 2, mRgba.rows() / 2);
                    cameraMatrix.put(2, 0, 0);
                    cameraMatrix.put(2, 1, 0);
                    cameraMatrix.put(2, 2, 1);

                    //distCoeffs
                    distCoeffs.put(0, 0, 0);
                    distCoeffs.put(0, 1, 0);
                    distCoeffs.put(0, 2, 0);
                    distCoeffs.put(0, 3, 0);
                    distCoeffs.put(0, 4, 0);

                    mGray.copyTo(estimateFrame);
                    //偵測CAMERA的keypoint and descriptor
                    featureDetector.detect(estimateFrame, keyPoint_test);
                    descriptorExtractor.compute(estimateFrame, keyPoint_test, descriptor2);

                    Log.i("descriptor1",""+descriptor1.toString());
                    Log.i("descriptor2",""+descriptor2.toString());
                    if(descriptor2.empty()){
                        DETECTTOMAKER = FALSE;
                        continue;
                    }

                    matcher.match(descriptor1, descriptor2, matches);
                    List<DMatch> matchesList = matches.toList();

                    //若沒有任何點match直接return,可以防止當matchesList沒有任何東西時OutOfBoundary的情況
                    if (matchesList.isEmpty()) {
                        DETECTTOMAKER = FALSE;
                        continue;
                    }
                    Double max_dist = 0.0;
                    Double min_dist = 100.0;
                    Log.i("descriptor", " row: " + descriptor1.rows() + " col: " + descriptor1.cols());
                    for (int i = 0; i < descriptor1.rows(); i++) {
                        Double dist = (double) matchesList.get(i).distance;
                        if (dist < min_dist) min_dist = dist;
                        if (dist > max_dist) max_dist = dist;
                    }
                    Log.i("distance", "min: " + min_dist + " max: " + max_dist);
                    if (min_dist > 60) {
                        DETECTTOMAKER = FALSE;
                        continue;
                    }
                    LinkedList<DMatch> good_matches = new LinkedList<DMatch>();
                    MatOfDMatch gm = new MatOfDMatch();
                    //對匹配結果進行篩選
                    for (int i = 0; i < descriptor1.rows(); i++) {
                        if (matchesList.get(i).distance <= 2 * min_dist) {
                            good_matches.addLast(matchesList.get(i));
                        }
                    }

                    Log.i("goodMatcheSize", " " + good_matches.size());
                    if (good_matches.size() < 4) {
                        DETECTTOMAKER = FALSE;
                        continue;
                    }
                    gm.fromList(good_matches);
                    List<KeyPoint> keypoints_objectList = keyPoint_train.toList();
                    List<KeyPoint> keypoints_sceneList = keyPoint_test.toList();
                    LinkedList<Point> objList = new LinkedList<Point>();
                    LinkedList<Point> sceneList = new LinkedList<Point>();
                    //將匹配到的特徵點取出
                    for (int i = 0; i < good_matches.size(); i++) {
                        //Log.i("point1",""+keypoints_objectList.get(good_matches.get(i).queryIdx).pt);
                        //Log.i("point2",""+keypoints_sceneList.get(good_matches.get(i).trainIdx).pt);
                        circle(mRgba,keypoints_sceneList.get(good_matches.get(i).trainIdx).pt,1,new Scalar(0,255,0),4);
                        objList.addLast(keypoints_objectList.get(good_matches.get(i).queryIdx).pt);
                        sceneList.addLast(keypoints_sceneList.get(good_matches.get(i).trainIdx).pt);
                    }

                    MatOfPoint2f obj = new MatOfPoint2f();
                    obj.fromList(objList);
                    MatOfPoint2f scene = new MatOfPoint2f();
                    scene.fromList(sceneList);
                    scene.copyTo(estimateScenePoint);

                    //推出相機在世界座標系的位置
                    List<Point3> makerList = new ArrayList<Point3>();
                    for (int i = 0; i < good_matches.size(); i++) {
                        //Log.i("XandY"," "+keypoints_objectList.get(good_matches.get(i).queryIdx).pt.x+" "+keypoints_objectList.get(good_matches.get(i).queryIdx).pt.y);
                        makerList.add(new Point3(-((keypoints_objectList.get(good_matches.get(i).queryIdx).pt.x - maker.cols() / 2) * 285 / maker.cols()),
                                0, ((keypoints_objectList.get(good_matches.get(i).queryIdx).pt.y - maker.rows() / 2) * 200 / maker.rows())));
                    }
                    MatOfPoint3f makerPoints = new MatOfPoint3f();
                    makerPoints.fromList(makerList);
                    makerPoints.copyTo(estimateMakerPoints);

                    Mat tempRvec = new Mat();
                    Rvec.copyTo(tempRvec);
                    Calib3d.solvePnPRansac(makerPoints, scene, cameraMatrix, distCoeffs, Rvec, Tvec);//CV_EPNP n>3
                    //Log.i("Tvec",""+Tvec.get(0,0)[0]+" "+Tvec.get(1,0)[0]+" "+Tvec.get(2,0)[0]);
                    //Rvec有3個參數要傳給server,代表相機跟標記間的角度關西
                    Log.i("Rvec", "" + Rvec.get(0, 0)[0] + " " + Rvec.get(1, 0)[0] + " " + Rvec.get(2, 0)[0]);
                    Log.i("Angle", "" + Rvec.get(0, 0)[0] * 180 / Math.PI + " " + Rvec.get(1, 0)[0] * 180 / Math.PI + " " + Rvec.get(2, 0)[0] * 180 / Math.PI);
                    //將rvec轉成矩陣
                    Mat rotMat = new Mat(3, 3, CvType.CV_32F);
                    Calib3d.Rodrigues(Rvec, rotMat);
                    //camera世界座標
                    Mat result = new Mat();
                    Core.gemm(rotMat.inv(), Tvec, -1, new Mat(), 0, result);//result=alpha*src1*src2+beta*src3
                    //Log.i("cameraWorld", result.get(0,0)[0]+" "+result.get(1,0)[0]+" "+result.get(2,0)[0]);
                    // right-handed coordinates system (OpenCV) to left-handed one (Unity)
                    if (result.get(0, 0)[0] < 1000 && result.get(0, 0)[0] > -1000 &&
                            result.get(1, 0)[0] < 0 && result.get(1, 0)[0] > -1000 &&
                            result.get(2, 0)[0] < 1000 && result.get(2, 0)[0] > -1000) {
                        Position = new Point3(result.get(0, 0)[0], result.get(1, 0)[0], result.get(2, 0)[0]);
                        UnityPosition = new Point3(Position.x, -Position.y, Position.z);
                    } else {
                        tempRvec.copyTo(Rvec);
                    }
                    estimateFrame.copyTo(lightFrame);
                    DETECTTOMAKER = TRUE;
                }
            }
        });
        positionEstimate.start();

        btnClose.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    checkConnect = false;
                    outputStream.close();
                    inputStream.close();
                    socket.close();
                }catch(IOException e){
                }
            }
        });

        btnReady.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                btnReady.setText(ready==false?"取消準備":"準備");
                ready = !ready;
                /*if(ready==0){
                    btnReady.setText("取消準備");
                    ready = 1;
                }
                else {
                    btnReady.setText("準備");
                    ready = 0;
                }*/
            }
        });

        btnSend.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                connectServer = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            // 創建socket IP port
                            socket = new Socket("140.121.197.164", 80);
                            // 判斷是否連接成功
                            try {
                                Thread.sleep(50);
                            } catch (InterruptedException ex) {
                                Thread.currentThread().interrupt();
                            }
                            checkConnect = true;
                            //傳送圖片解析度
                            st = mRgba.cols() + " " + mRgba.rows();
                            outputStream = socket.getOutputStream();
                            outputStream.write((st).getBytes("utf-8"));
                            outputStream.flush();

                            // record player's list
                            InputStream in = socket.getInputStream();
                            playerList = in.read();
                            Log.i("playerlist", "playerlist : "+playerList);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                });
                transmission =  new Thread(new Runnable() {
                    @Override
                    public void run() {
                        long time1, time2, time3, time4, time5;
                        while (checkConnect) {
                            //將資訊串接成string傳送
                            st = decimalFormat.format(UnityPosition.x / 10 * 6.5) + " " + decimalFormat.format(UnityPosition.y / 10 * 6.5) + " " + decimalFormat.format(UnityPosition.z / 10 * 6.5) + " " + decimalFormat.format(Rvec.get(0, 0)[0]) + " " + decimalFormat.format(Rvec.get(1, 0)[0]) + " " + decimalFormat.format(Rvec.get(2, 0)[0]) + " " + playerList + " " + move + " " + ready + " ";
                            try {
                                /*將字串寫入輸出流------------------------------------------------*/
                                //從socket獲得輸出流outputStream
                                time1 = System.currentTimeMillis();
                                outputStream = socket.getOutputStream();
                                //寫入數據到輸出流，st為參數字串
                                outputStream.write((st).getBytes("utf-8"));
                                //發送
                                outputStream.flush();
                                //重置move
                                move = -1;
                                time3 = System.currentTimeMillis();
                                /*接收檔案--------------------------------------------------------*/
                                inputStream = socket.getInputStream();
                                datasize = 0;
                                while(datasize == 0) {datasize = inputStream.available();}//保證數據有收到
                                //datasize = inputStream.available();
                                Log.i("接收大小", "" + datasize);
                                data = new byte[datasize];
                                time4 = System.currentTimeMillis();
                                if (datasize > 1451) {
                                    //buffer = java.lang.Math.abs(buffer - 1);
                                    //data[buffer] = new byte[datasize];
                                    /*將緩衝區read到data------------------------------------------*/
                                    int len = 0;
                                    while(len<datasize){//保證讀取接收到的量
                                        len += inputStream.read(data, len, datasize-len);
                                    }
                                    //inputStream.read(data[buffer], 0, datasize);
                                    /*將data轉為所需型態------------------------------------------*/
                                    lock.lock();
                                    bmp = BitmapFactory.decodeByteArray(data, 0, data.length);
                                    paste = Imgcodecs.imdecode(new MatOfByte(data), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
                                    Message msg = Message.obtain();
                                    mMainHandler.sendMessage(msg);
                                    Log.i("resolution", paste.toString());
                                    if (paste.empty() != true) {
                                        Imgproc.cvtColor(paste, paste, COLOR_RGB2BGRA);
                                    }
                                    lock.unlock();

                                }else if(datasize == 5){
                                    /*接收場上訊息------------------------------------------------*/
                                    final int bufferSize = 1024;
                                    final char[] buffer = new char[bufferSize];
                                    final StringBuilder out = new StringBuilder();
                                    Reader in = new InputStreamReader(inputStream, "UTF-8");
                                    int rsz = in.read(buffer, 0, buffer.length);
                                    /*if (rsz < 0)
                                        break;*/
                                    out.append(buffer, 0, rsz);
                                    st1 = out.toString();
                                    Log.i("st1",""+st1 + "total:" + datasize);
                                    //setStatus(st1);
                                    Message msg = Message.obtain();
                                    mMainHandler.sendMessage(msg);
                                    /*------------------------------------------------------------*/
                                }else{
                                    inputStream.read();
                                }
                                time2 = System.currentTimeMillis();
                                //Log.i("720*480 發送到接收花了：", "" + (time2-time1) + "毫秒");
                                //Log.i("2.doSomething()花了：", "" + (time3-time1)+"size" + datasize);
                                //Log.i("3.doSomething()花了：", "" + (time4-time3)+"size" + datasize);
                                //Log.i("4.doSomething()花了：", "" + (time2-time4)+"size" + datasize);
                                try {
                                    Thread.sleep(150);
                                } catch (InterruptedException ex) {
                                    Thread.currentThread().interrupt();
                                }
                            } catch (IOException e) {
                                e.printStackTrace();
                            }
                        }
                    }
                });
                //start thread
                connectServer.start();
                try {
                    connectServer.join();
                    transmission.start();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });

        mGesture = new GestureDetector(this, new GestureDetector.SimpleOnGestureListener() {
            @Override
            public boolean onSingleTapUp(MotionEvent e) {
                // TODO Auto-generated method stub
                return false;
            }

            @Override
            public void onShowPress(MotionEvent e) {
                // TODO Auto-generated method stub

            }

            @Override
            public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY) {
                // TODO Auto-generated method stub
                return false;
            }

            @Override
            public void onLongPress(MotionEvent e) {
                // TODO Auto-generated method stub

            }

            @Override
            public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {
                if (e1.getX() - e2.getX() > FLIP_DISTANCE) {
                    Log.i(Tag, "向左滑...");
                    move = 3;
                    Log.i(Tag, e1.getX() + " " + e1.getY());
                    Log.i(Tag, e2.getX() + " " + e2.getY());
                    return true;
                }
                if (e2.getX() - e1.getX() > FLIP_DISTANCE) {
                    Log.i(Tag, "向右滑...");
                    move = 2;
                    Log.i(Tag, e1.getX() + " " + e1.getY());
                    Log.i(Tag, e2.getX() + " " + e2.getY());
                    return true;
                }
                if (e1.getY() - e2.getY() > FLIP_DISTANCE) {
                    Log.i(Tag, "向上滑...");
                    move = 0;
                    Log.i(Tag, e1.getX() + " " + e1.getY());
                    Log.i(Tag, e2.getX() + " " + e2.getY());
                    return true;
                }
                if (e2.getY() - e1.getY() > FLIP_DISTANCE) {
                    Log.i(Tag, "向下滑...");
                    move = 1;
                    Log.i(Tag, e1.getX() + " " + e1.getY());
                    Log.i(Tag, e2.getX() + " " + e2.getY());
                    return true;
                }
                return false;
            }

            @Override
            public boolean onDown(MotionEvent e) {
                // TODO Auto-generated method stub
                return false;
            }
        });


        /*-------------------------------------------------------------------------------------------*/
        //threadpool 1.降低資源消耗 重複利用已建立線程 2.高度線程管理
        mThreadPool = Executors.newCachedThreadPool();
        mMainHandler = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                imgView.setImageBitmap(bmp);
                char[] c  = st1.toCharArray();
                switch(c[0]){
                    case '0':
                        txvcolor.setText("紅色");
                        break;
                    case '1':
                        txvcolor.setText("黃色");
                        break;
                    case '2':
                        txvcolor.setText("藍色");
                        break;
                    case '3':
                        txvcolor.setText("綠色");
                        break;
                    default:
                        txvcolor.setText("顏色");
                }
                txvspeed.setText("速度"+c[1]);
                txvsize.setText("大小:"+c[2]);
                switch(c[3]){
                    case '0':
                        txvalive.setText("落敗");
                        break;
                    case '1':
                        txvalive.setText("存活");
                        break;
                    default:
                        txvalive.setText("未開始");
                }
                /*switch (msg.what) {
                    case 0:
                        //receive_message.setText(input);
                        break;
                }*/
            }
        };
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        return mGesture.onTouchEvent(event);
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
        mRgba = new Mat(height, width, CV_8UC4);//CV_(位元數)+(資料型態)+(Channel數)
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
        mGray = inputFrame.gray();
        MatOfPoint2f nextPtr = new MatOfPoint2f();


        if (DETECTTOMAKER) {
            /*Video.calcOpticalFlowPyrLK(lightFrame, mGray, estimateScenePoint, nextPtr, statusMat, errSimilarityMat);
            Log.i("featureSize"," "+estimateMakerPoints.rows()+" "+nextPtr.rows());
            if(estimateMakerPoints.rows()!=nextPtr.rows())
                return mRgba;
            Mat tempRvec=new Mat();
            Rvec.copyTo(tempRvec);
            Calib3d.solvePnPRansac(estimateMakerPoints, nextPtr, cameraMatrix, distCoeffs, Rvec, Tvec);//CV_EPNP n>3
            //將rvec轉成矩陣
            Mat rotMat=new Mat(3,3,CvType.CV_32F);
            Calib3d.Rodrigues(Rvec,rotMat);
            Mat result=new Mat();
            Core.gemm(rotMat.inv(),Tvec,-1,new Mat(),0,result);//result=alpha*src1*src2+beta*src3
            // right-handed coordinates system (OpenCV) to left-handed one (Unity)
            if(result.get(0, 0)[0] < 1000 && result.get(0, 0)[0]> -1000 &&
                    result.get(1, 0)[0] < 0 && result.get(1, 0)[0] > -1000 &&
                    result.get(2, 0)[0] < 1000 && result.get(2, 0)[0] > -1000) {
                Position = new Point3(result.get(0, 0)[0], result.get(1, 0)[0], result.get(2, 0)[0]);
                UnityPosition = new Point3(Position.x, -Position.y, Position.z);
            }else{
                tempRvec.copyTo(Rvec);
            }*/
            Log.i("lightflowTracking", decimalFormat.format(Position.x) + " " + decimalFormat.format(Position.y) + " "
                    + decimalFormat.format(Position.z));
            Mat frame = new Mat();
            Log.i("paste2", paste.toString());
            if (paste.rows() == mRgba.rows() && paste.cols() == mRgba.cols()) {
                lock.lock();
                paste.copyTo(pasteBuffer);
                Log.i("paste", paste.toString());
                //轉灰階
                Imgproc.cvtColor(paste, pasteGray, COLOR_BGR2GRAY);
                //大于阈值部分被置为0，小于部分被置为255 取得mask
                Imgproc.threshold(pasteGray, pasteGray, 0, 255, Imgproc.THRESH_BINARY_INV);
                Core.bitwise_and(mRgba, mRgba, frame, pasteGray);
                Core.add(frame, paste, frame);
                //paste = new Mat();
                lock.unlock();
                return frame;
            } else if (pasteBuffer.empty() != true) {
                //轉灰階
                try {
                    Imgproc.cvtColor(pasteBuffer, pasteGray, COLOR_BGR2GRAY);
                    //大于阈值部分被置为0，小于部分被置为255 取得mask
                    Imgproc.threshold(pasteGray, pasteGray, 0, 255, Imgproc.THRESH_BINARY_INV);
                    Core.bitwise_and(mRgba, mRgba, frame, pasteGray);
                    Core.add(frame, pasteBuffer, frame);
                    return frame;
                } catch (IllegalArgumentException e) {
                    Log.i("BitmapE", "" + e);
                }
            }
        }
        /*nextPtr.copyTo(estimateScenePoint);
        mGray.copyTo(lightFrame);*/
        mGray.copyTo(frameBuffer);
        return mRgba;
    }
}