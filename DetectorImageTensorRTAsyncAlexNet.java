package org.cqels.objdetection.async;


import org.apache.commons.lang3.ArrayUtils;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.cqels.common.math.array.FloatArray;
import org.cqels.common.objdetection.DetectedObjectImage;
import org.cqels.common.objdetection.context.DetectorImageContext;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

public class DetectorImageTensorRTAsyncAlexNet extends DetectorImageTensorRTAsync {


    static String[] inputTensorName = new String[]{"data_0"};
    static String outputTensorNames = "prob_1";
    float pixelMean[] = {0.485f, 0.456f, 0.406f};
    float pixelStd[] = {0.229f, 0.224f, 0.225f};


    public DetectorImageTensorRTAsyncAlexNet(DetectorImageContext detectorImageContext) {
        super(detectorImageContext);
        System.out.println("Success created class");
    }


    @Override
    protected Map<String, Pointer> processInput(byte[] input, int batchSize) {

        Map<String, Pointer> inputBindings = new HashMap<>();
        int inputH = this.getInputHeight();
        int inputW = this.getInputWidth();
        int inputC = this.getInputDepth();

        int volImg = inputH * inputW * inputC;
        int volChl = inputH * inputW;

        float[] data = new float[inputH * inputW * inputC * batchSize];
        for (int batch = 0; batch < batchSize; ++batch) {
            for (int c = 0; c < inputC; ++c) {
                for (int j = 0; j < volChl; ++j) {
                    int inputIdx = j * inputC + (2 - c);
                    int dataIdx = batch * volImg + c * volChl + j;
                    data[dataIdx] = (float) (input[inputIdx] & 0xFF) /255 - pixelMean[c] / pixelStd[c] ;
                }
            }
        }

        FloatBuffer dataBuffer = FloatBuffer.wrap(data);
        Pointer dataPointer = new FloatPointer(dataBuffer);
        inputBindings.put(inputTensorName[0], dataPointer);

        System.out.println("Process input completed");

        inputBindings.put(inputTensorName[0], dataPointer);

        return inputBindings;
    }


    @Override
    protected Map<String, Pointer> processInput(byte[][] input) {
        Map<String, Pointer> inputBindings = new HashMap<>();
        int inputH = this.getInputHeight();
        int inputW = this.getInputWidth();
        int inputC = this.getInputDepth();
        int batchSize = input.length;

        int volImg = inputH * inputW * inputC;
        int volChl = inputH * inputW;

        float[] data = new float[inputH * inputW * inputC * batchSize];
        for (int batch = 0; batch < batchSize; ++batch) {
            for (int c = 0; c < inputC; ++c) {
                for (int j = 0; j < volChl; ++j) {
                    int inputIdx = j * inputC + (2 - c);
                    int dataIdx = batch * volImg + c * volChl + j;
                    data[dataIdx] = (float) (input[batch][inputIdx] & 0xFF) /255 - pixelMean[c] / pixelStd[c] ;
                }
            }
        }

        FloatBuffer dataBuffer = FloatBuffer.wrap(data);
        Pointer dataPointer = new FloatPointer(dataBuffer);
        inputBindings.put(inputTensorName[0], dataPointer);

        System.out.println("Process input completed");

        return inputBindings;
    }


    @Override
    protected DetectedObjectImage[][] processOutput(Map<String, Pointer> output, int batchSize) {

        float[] out = FloatArray.toFloatArray(output.get(outputTensorNames).asByteBuffer().asFloatBuffer());
        System.out.println(out.length);
        float[][] out2 = FloatArray.split(out, batchSize);

        for (int b=0; b<batchSize; b++){
            System.out.print("Output of image: " + b + " ");
            System.out.println(ArrayUtils.toString(out2[b]));
        }

        return null;
    }
}
