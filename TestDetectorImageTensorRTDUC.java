package org.cqels.objdetection;

import junit.framework.TestCase;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.cqels.common.objdetection.DetectedObjectImage;
import org.cqels.common.objdetection.DetectorImage;
import org.cqels.common.objdetection.context.DetectorImageContext;
import org.cqels.common.objdetection.context.DetectorImageContextONNX;
import org.cqels.common.opencv.render.Draw;
import org.cqels.common.runtime.SystemContext;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.cqels.common.runtime.SystemContext.WORK_SPACE;
import static org.cqels.unittest.UnitTest.mat2byteArray;
import static org.cqels.unittest.UnitTest.matImgs;

public class TestDetectorImageTensorRTDUC extends TestCase {
    private static String detectorName = "DUC";
    private static int batchSize = 8;

    public void _test_register_DUC(){
        DetectorImageContext.InputImageSize inputImageSize = new DetectorImageContext.InputImageSize();
        inputImageSize.setDepth(3);
        inputImageSize.setHeight(800);
        inputImageSize.setWidth(800);

        DetectorImageContextONNX detectorImageContextONNX = new DetectorImageContextONNX();
        detectorImageContextONNX.setDetectorName(detectorName);
        detectorImageContextONNX.setBatchSize(32);
        detectorImageContextONNX.setPathToONNX(WORK_SPACE + "detector/onnx/ResNet101-DUC-7.onnx");
        detectorImageContextONNX.setClassPath("org.cqels.objdetection.async.DetectorImageTensorRTAsyncDUC");

        detectorImageContextONNX.setInputImageSize(inputImageSize);

        DetectorImageTensorRTRegister detectorImageTensorRTRegister = new DetectorImageTensorRTRegister();
        DetectorImageTensorRTContext detectorImageTensorRTContext = detectorImageTensorRTRegister.registerV2(detectorImageContextONNX);
    }


    public void test_detection_DUC() {
        DetectorImageTensorRTManager detectorImageTensorRTManager = new DetectorImageTensorRTManager();
        DetectorImage detectorImage = detectorImageTensorRTManager.findDetectorImageByName(detectorName);

        byte[][] inputs = new byte[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            Mat mat = matImgs[i];
            resize(mat, mat, new Size(detectorImage.getInputWidth(), detectorImage.getInputHeight()));
            inputs[i] = mat2byteArray(mat);
        }

        DetectedObjectImage[][] dois = detectorImage.detectBatch(inputs);

        for (int i = 0; i < batchSize; i++) {

            if (dois[i] != null) {
                Draw.render(matImgs[i], dois[i], "test", 1000);
            } else {
                Draw.render(matImgs[i], new DetectedObjectImage[0], "test", 1000);
            }

            System.out.println(WORK_SPACE + "image/" + detectorName + "/" + i + ".jpg");
            imwrite(WORK_SPACE + "image/" + detectorName + "/" + i + ".jpg", matImgs[i]);
        }
    }
}
