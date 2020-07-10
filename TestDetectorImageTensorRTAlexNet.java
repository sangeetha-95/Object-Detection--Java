package org.cqels.objdetection;

import junit.framework.TestCase;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.cqels.common.objdetection.DetectorImage;
import org.cqels.common.objdetection.context.DetectorImageContext;
import org.cqels.common.objdetection.context.DetectorImageContextONNX;
import org.cqels.common.runtime.SystemContext;

import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.cqels.unittest.UnitTest.mat2byteArray;
import static org.cqels.unittest.UnitTest.matImgs;

public class TestDetectorImageTensorRTAlexNet extends TestCase {
    private static String detectorName = "AlexNet";
    private static int batchSize = 8;

    public void test_register_alexnet() {
        DetectorImageContext.InputImageSize inputImageSize = new DetectorImageContext.InputImageSize();
        inputImageSize.setWidth(224);
        inputImageSize.setHeight(224);
        inputImageSize.setDepth(3);

        DetectorImageContextONNX detectorImageContextONNX = new DetectorImageContextONNX();
        detectorImageContextONNX.setDetectorName(detectorName);
        detectorImageContextONNX.setBatchSize(32);
        detectorImageContextONNX.setPathToONNX(SystemContext.USER_HOME + "experiment/detector/onnx/bvlcalexnet-7.onnx");
        detectorImageContextONNX.setClassPath("org.cqels.objdetection.async.DetectorImageTensorRTAsyncAlexNet");

        detectorImageContextONNX.setInputImageSize(inputImageSize);

        DetectorImageTensorRTRegister detectorImageTensorRTRegister = new DetectorImageTensorRTRegister();
        DetectorImageTensorRTContext detectorImageTensorRTContext = detectorImageTensorRTRegister.registerV2(detectorImageContextONNX);
    }


    public void test_detection_alexnet() {
        DetectorImageTensorRTManager detectorImageTensorRTManager = new DetectorImageTensorRTManager();
        DetectorImage detectorImage = detectorImageTensorRTManager.findDetectorImageByName(detectorName);

        byte[][] inputs = new byte[batchSize][];

        for (int i = 0; i < batchSize; i++) {
            Mat mat = matImgs[i];
            resize(mat, mat, new Size(detectorImage.getInputWidth(), detectorImage.getInputHeight()));
            inputs[i] = mat2byteArray(mat);
        }

        detectorImage.detectBatch(inputs);


    }

}
