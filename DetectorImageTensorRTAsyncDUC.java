package org.cqels.objdetection.async;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.cqels.common.math.array.FloatArray;
import org.cqels.common.math.array.IntArray;
import org.cqels.common.objdetection.Box;
import org.cqels.common.objdetection.DetectedObjectImage;
import org.cqels.common.objdetection.context.DetectorImageContext;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;


public class DetectorImageTensorRTAsyncDUC extends DetectorImageTensorRTAsync {
    private static final String[] inputTensorNames = new String[]{"data"};
    private static final String[] outputTensorNames = new String[]{"seg_loss"};

    private final float[] pixelMean = new float[]{102.9801f, 115.9465f, 122.7717f};

    private final String[] classes = new String[]{
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv monitor"
    };


    public DetectorImageTensorRTAsyncDUC(DetectorImageContext detectorImageContext) {
        super(detectorImageContext);
    }

    @Override
    protected Map<String, Pointer> processInput(byte[][] input) {
        Map<String, Pointer> inputBindings = new HashMap<>();

        int h = getInputHeight();
        int w = getInputWidth();
        int ch = getInputDepth();
        int volImg = h * w * ch;
        int volChl = h * w;

        int batchSize = input.length;

        float[] data = new float[h * w * ch * batchSize];
        for (int batch = 0; batch < batchSize; batch++) {
            for (int c = 0; c < ch; c++) {
                for (int j = 0; j < volChl; j++) {
                    int inputIdx = j * ch + (2 - c);
                    int dataIdx = batch * volImg + c * volChl + j;
                    data[dataIdx] = (float) (input[batch][inputIdx] & 0xFF) - pixelMean[c];
                }
            }
        }

        FloatBuffer dataBuffer = FloatBuffer.wrap(data);
        Pointer dataPointer = new FloatPointer(dataBuffer);
        inputBindings.put(inputTensorNames[0], dataPointer);

        System.out.println("Process input");

        return inputBindings;
    }

    @Override
    protected Map<String, Pointer> processInput(byte[] input, int batchSize) {
        Map<String, Pointer> inputBindings = new HashMap<>();

        int h = getInputHeight();
        int w = getInputWidth();
        int ch = getInputHeight();
        int volImg = h * w * ch;
        int volChl = h * w;

        float[] data = new float[h * w * ch * batchSize];
        for (int batch = 0; batch < batchSize; batch++) {
            for (int c = 0; c < ch; c++) {
                for (int j = 0; j < volChl; j++) {
                    int inputIdx = batch * volImg + j * ch + (2 - c);
                    int dataIdx = batch * volImg + c * volChl + j;
                    data[dataIdx] = (float) (input[inputIdx] & 0xFF) - pixelMean[c];
                }
            }
        }

        FloatBuffer dataBuffer = FloatBuffer.wrap(data);
        Pointer dataPointer = new FloatPointer(dataBuffer);
        inputBindings.put(inputTensorNames[0], dataPointer);

        System.out.println("Process input");

        return inputBindings;

    }

    @Override
    protected DetectedObjectImage[][] processOutput(Map<String, Pointer> output, int batchSize) {
        DetectedObjectImage[][] result = new DetectedObjectImage[batchSize][];

        float[] output_1 = FloatArray.toFloatArray(output.get(outputTensorNames[0]).asByteBuffer().asFloatBuffer());
        System.out.println("Process output" + " " + output_1.length + "\n");

        for (int b = 0; b < batchSize; b++) {

            ArrayList<DetectedObjectImage> doiList = new ArrayList<>();

            for (int count = 0; count < output_1[b]; count++) {
                int offset = (b * 200 + count) * 7;
                String label = classes[(int) output_1[1]];
                float conf = output_1[offset + 2] * 100.f;
                float left = output_1[offset + 3] * getInputWidth();
                float top = output_1[offset + 4] * getInputHeight();
                float right = output_1[offset + 5] * getInputWidth();
                float bottom = output_1[offset + 6] * getInputHeight();

                Box box = new Box().setBoxTopLeftBottomRight(top, left, bottom, right);

                DetectedObjectImage doi = constructDetectedObject(label, conf, box);
                doiList.add(doi);

                System.out.println("Confidence" + " " + conf  + "\n");

            }

            DetectedObjectImage[] doiArr = new DetectedObjectImage[doiList.size()];
            doiList.toArray(doiArr);
            result[b] = doiArr;

        }

        return result;
    }
}
