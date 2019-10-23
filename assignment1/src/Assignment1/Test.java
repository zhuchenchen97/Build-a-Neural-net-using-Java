package Assignment1;

import java.io.IOException;
import java.util.Arrays;

public class Test {
    private int argNumInputs = 2;
    private int argNumHidden = 4;
    private int argNumOutputs = 1;
    private int argNumTrainingSet = 4;
    private double argLearningRate = 0.2;
    private double argMomentumTerm = 0.9;
    private double bias = 1;
    private boolean binary = false;
    private double argA;
    private double argB;
    private double errorThreshold = 0.05;
    
    private double[][] inputX = new double[argNumTrainingSet][argNumInputs+1]; //plus one bias value
	private double[][] outputY = new double[argNumTrainingSet][argNumOutputs];

	public void initializeTrainingSet() {
	 if(binary) {
		 argA = 0;
		 argB = 1;
		 inputX[0][0]=bias;
		 inputX[0][1]=0;
		 inputX[0][2]=0;
		 
		 inputX[1][0]=bias;
		 inputX[1][1]=0;
		 inputX[1][2]=1;
		 
		 inputX[2][0]=bias;
		 inputX[2][1]=1;
		 inputX[2][2]=0;
		 
		 inputX[3][0]=bias;
		 inputX[3][1]=1;
		 inputX[3][2]=1;
		 
		 outputY[0][0]=0;
		 outputY[1][0]=1;
		 outputY[2][0]=1;
		 outputY[3][0]=0;
	 }else {
		 argA = -1;
		 argB = 1;
		 inputX[0][0]=bias;
		 inputX[0][1]=-1;
		 inputX[0][2]=-1;
		 
		 inputX[1][0]=bias;
		 inputX[1][1]=-1;
		 inputX[1][2]=1;
		 
		 inputX[2][0]=bias;
		 inputX[2][1]=1;
		 inputX[2][2]=-1;
		 
		 inputX[3][0]=bias;
		 inputX[3][1]=1;
		 inputX[3][2]=1;
		 
		 outputY[0][0]=-1;
		 outputY[1][0]=1;
		 outputY[2][0]=1;
		 outputY[3][0]=-1;
	 }
    }

	public void runNeuralNet() throws IOException {
 		 int aveEpochNum=0;
		 int trials=500;
		 int maxEpochNum=0;
		 int minEpochNum=10000;
	   for(int i=0;i<trials;i++) {        
		 initializeTrainingSet();
		 // System.out.println(Arrays.deepToString(inputX));
		 NeuralNet testNeuronNet = new NeuralNet(argNumInputs,argNumHidden,argNumOutputs,argNumTrainingSet,argLearningRate,argMomentumTerm,argA,argB);
		 testNeuronNet.buildLayers();
		 testNeuronNet.allLayers();
		 testNeuronNet.initializeWeights();

	 
            testNeuronNet.runNeuralNet(errorThreshold,inputX,outputY);
            if(testNeuronNet.totalEpochNum>maxEpochNum) {
            	maxEpochNum = testNeuronNet.totalEpochNum;
            }
            if(testNeuronNet.totalEpochNum<minEpochNum) {
            	minEpochNum = testNeuronNet.totalEpochNum;
            }
            aveEpochNum=aveEpochNum+testNeuronNet.totalEpochNum;
		 testNeuronNet.printRunResults(testNeuronNet.getErrorArray(),"F://502result//bipolar-0.9//result"+i+".csv");			 
		 }
		 aveEpochNum = aveEpochNum/trials;
		 System.out.println("ave:"+aveEpochNum);
		 System.out.println("max:"+maxEpochNum);
		 System.out.println("min:"+minEpochNum);
		
	}
	
	public static void main(String[] args) throws IOException {
		 Test test = new Test();
		 test.runNeuralNet();
	 }
  
}
