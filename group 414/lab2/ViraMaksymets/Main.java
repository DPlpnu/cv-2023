package com.company;

import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;


import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;


public class Main {
    static double threshold = 0.5;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) throws IOException {
        Mat scene = Highgui.imread("src/com/company/contrast.jpg", Highgui.CV_LOAD_IMAGE_GRAYSCALE);
        if (scene.empty()) {
            throw new Error("scene empty");
        }

        Mat rotated = rotate(matToBufferedImage(scene));
        if (rotated.empty()) {
            throw new Error("rotated empty");
        }

        //openCVMatcher(scene, rotated);
        myCustomMatcher(scene, rotated);
    }

    private static void openCVMatcher(Mat sceneImageMat, Mat objectImageMat) {
        FeatureDetector surf = FeatureDetector.create(FeatureDetector.SURF);
        DescriptorExtractor surfExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);

        Mat objectDescriptor = new Mat();
        MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();
        surf.detect(objectImageMat, objectKeyPoints);
        surfExtractor.compute(objectImageMat, objectKeyPoints, objectDescriptor);
        System.out.println("Number of object keypoints: " + objectKeyPoints.size());

        Mat sceneDescriptor = new Mat();
        MatOfKeyPoint sceneKeyPoints = new MatOfKeyPoint();
        surf.detect(sceneImageMat, sceneKeyPoints);
        surfExtractor.compute(sceneImageMat, sceneKeyPoints, sceneDescriptor);
        System.out.println("Number of scene keypoints: " + sceneKeyPoints.size());

        MatOfDMatch matches = new MatOfDMatch();
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_SL2);
        matcher.match(sceneDescriptor, objectDescriptor, matches);
        System.out.println("Number of matches: " + matches.size());

        LinkedList<DMatch> goodMatches = new LinkedList<>();
        MatOfDMatch gm = new MatOfDMatch();
        for (int i = 0; i < objectDescriptor.rows(); i++) {
            if (matches.toList().get(i).distance < threshold) {
                goodMatches.addLast(matches.toList().get(i));
            }
        }
        gm.fromList(goodMatches);
        System.out.println("Good matches " + gm.size());

        Mat outImg = new Mat(sceneImageMat.rows() * 2, sceneImageMat.cols() * 2, CvType.CV_8UC3);
        Features2d.drawMatches(objectImageMat, objectKeyPoints, sceneImageMat, sceneKeyPoints, gm, outImg,
                new Scalar(255, 0, 0), new Scalar(255, 0, 0), new MatOfByte(), Features2d.NOT_DRAW_SINGLE_POINTS);
        display(matToBufferedImage(outImg));
    }


    private static void myCustomMatcher(Mat sceneImageMat, Mat objectImageMat) {
        FeatureDetector surf = FeatureDetector.create(FeatureDetector.SURF);
        DescriptorExtractor surfExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);

        Mat objectDescriptor = new Mat();
        MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();
        surf.detect(objectImageMat, objectKeyPoints);
        surfExtractor.compute(objectImageMat, objectKeyPoints, objectDescriptor);
        System.out.println("Number of object keypoints: " + objectKeyPoints.size());

        Mat sceneDescriptor = new Mat();
        MatOfKeyPoint sceneKeyPoints = new MatOfKeyPoint();
        surf.detect(sceneImageMat, sceneKeyPoints);
        surfExtractor.compute(sceneImageMat, sceneKeyPoints, sceneDescriptor);
        System.out.println("Number of scene keypoints: " + sceneKeyPoints.size());


        // Match descriptors
        MatOfDMatch matches = new MatOfDMatch();

        java.util.List<DMatch> matchesList = new ArrayList<>();
        for (int i = 0; i < objectDescriptor.rows(); i++) {
            double minDistance = Double.MAX_VALUE;
            int minIndex = 0;
            for (int j = 0; j < sceneDescriptor.rows(); j++) {
                double distance = 0;
                for (int k = 0; k < objectDescriptor.cols(); k++) {
                    double diff = objectDescriptor.get(i, k)[0] - sceneDescriptor.get(j, k)[0];
                    distance += Math.sqrt(diff * diff);
                }
                if (distance < minDistance) {
                    minDistance = distance;
                    minIndex = j;
                }
            }
            matchesList.add(new DMatch(i, minIndex, (float) minDistance));
        }
        System.out.println("Number of matches: " + matchesList.size());

        matchesList.subList(15, matchesList.size()).clear();
        matches.fromList(matchesList);

        Mat outImg = new Mat(sceneImageMat.rows() * 2, sceneImageMat.cols() * 2, CvType.CV_8UC3);
        Features2d.drawMatches(objectImageMat, objectKeyPoints, sceneImageMat, sceneKeyPoints, matches, outImg,
                new Scalar(255, 0, 0), new Scalar(255, 0, 0), new MatOfByte(), Features2d.NOT_DRAW_SINGLE_POINTS);

        display(matToBufferedImage(outImg));
    }

    private static Mat rotate(BufferedImage originalImg) throws IOException {
        BufferedImage image = originalImg.getSubimage(90, 90, 180, 180);
        double rotationRequired = Math.toRadians(25);
        double locationX = image.getWidth() / 2;
        double locationY = image.getHeight() / 2;
        AffineTransform tx = AffineTransform.getRotateInstance(rotationRequired, locationX, locationY);
        AffineTransformOp op = new AffineTransformOp(tx, AffineTransformOp.TYPE_BILINEAR);

        BufferedImage after = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        after = op.filter(image, after);

        return bufferedImageToMat(after);
    }

    public static Mat bufferedImageToMat(BufferedImage bi) throws IOException {
        // задання нового розміру
        int newWidth = 110;
        int newHeight = 110;

        // створення нового зображення з новим розміром
        BufferedImage resized = new BufferedImage(newWidth, newHeight, bi.getType());

        // зміна розміру зображення
        Graphics2D g = resized.createGraphics();
        g.drawImage(bi, 0, 0, newWidth, newHeight, null);
        g.dispose();

        File outputfile = new File("src/com/company/saved.jpg");
        ImageIO.write(resized, "jpg", outputfile);
        Mat mat = Highgui.imread("src/com/company/saved.jpg", Highgui.CV_LOAD_IMAGE_GRAYSCALE);
        return mat;
    }

    public static BufferedImage matToBufferedImage(Mat m) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (m.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = m.channels() * m.cols() * m.rows();
        byte[] b = new byte[bufferSize];
        m.get(0, 0, b); // get all the pixels
        BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }

    private static void display(BufferedImage img) {
        ImageIcon icon = new ImageIcon(img);
        JFrame frame = new JFrame();
        frame.setLayout(new FlowLayout());
        frame.setSize(img.getWidth() + 20, img.getHeight() + 45);
        JLabel lbl = new JLabel();
        lbl.setIcon(icon);
        frame.add(lbl);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

}
