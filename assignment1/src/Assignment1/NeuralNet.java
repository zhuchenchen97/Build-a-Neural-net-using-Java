package Assignment1;

import java.io.Console;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.omg.CORBA.PRIVATE_MEMBER;
import org.omg.CORBA.PUBLIC_MEMBER;

import Sarb.NeuralNetInterface;

public class NeuralNet implements NeuralNetInterface {
	
     static double bias = 1.0;
     private int argNumInputs;
     private int argNumHidden;
     private int argNumOutputs;
     private int argNumTrainingSet;
     private double argLearningRate;
     private double argMomentumTerm;
     private double argA;
     private double argB;

     
     private ArrayList<Neuron> inputLayer = new ArrayList<Neuron>();
     private ArrayList<Neuron> hiddenLayer = new ArrayList<Neuron>();
     private ArrayList<Neuron> outputLayer = new ArrayList<Neuron>();
     private ArrayList<ArrayList<Neuron>> allLayerArrayList = new ArrayList<ArrayList<Neuron>>();
     private ArrayList<Double> totalErrorInEachEpoch = new ArrayList<Double>(); //save the total error in each epoch
     public int totalEpochNum=1;
     private Neuron biasNeuron = new Neuron("bias",0,1);
     
     public void allLayers() {
    	 allLayerArrayList.add(hiddenLayer);
    	 allLayerArrayList.add(outputLayer);
     }
     public NeuralNet(int argNumInputs, int argNumHidden, int argNumOutputs, int argNumTrainingSet, double argLearningRate, double argMomentumTerm, double argA,
			double argB) {
		this.argNumInputs = argNumInputs;
		this.argNumHidden = argNumHidden;
		this.argNumOutputs = argNumOutputs;
		this.argNumTrainingSet = argNumTrainingSet;
		this.argLearningRate = argLearningRate;
		this.argMomentumTerm = argMomentumTerm;
		this.argA = argA;
		this.argB = argB;
		
		//this.initializeTrainingSet();


	}
     
     public void buildLayers() {
    	 //build input layer
    	 for(int i=0; i<argNumInputs; i++) {
    		 String id = "inputLayerNeuron" + Integer.toString(i);
    		 Neuron e = new Neuron(id,argA,argB);
    		 inputLayer.add(e);
    	 }
    	 //build hidden layer
    	 for(int i=0; i<argNumHidden; i++) {
    		 String id = "hiddenLayerNeuron" +Integer.toString(i);
    		 Neuron e = new Neuron(id,"customSigmoid",inputLayer,biasNeuron,argA,argB);
    		 hiddenLayer.add(e);
    	 }
    	 //build output layer
    	 for(int i=0; i<argNumOutputs; i++) {
    		 String id = "outputLayerNeuron" +Integer.toString(i);
    		 Neuron e = new Neuron(id,"customSigmoid",hiddenLayer,biasNeuron,argA,argB);
    		 outputLayer.add(e);
    	 }
    	 biasNeuron.setNeuronOut(1.0);
     }
     
     public double getWeightRandom(double lowerbound, double upperbound) {
    	 Random random = new Random();
    	 double weight = random.nextDouble() * (upperbound-lowerbound) + lowerbound;
    	 return weight;
     }

     public void initializeWeights() {
    	 double lowerbound = -0.5;
    	 double upperbound = 0.5;
    	 for(ArrayList<Neuron> al: allLayerArrayList) {
    		 for(Neuron neuron: al) {
    		 ArrayList<Edge> edges = neuron.getInEdges();
    		 for(Edge currentedge: edges) {
    			 currentedge.setWeight(getWeightRandom(lowerbound,upperbound));
    		 }
    		 Edge edge = neuron.getBiasEdge();
    		 edge.setWeight(getWeightRandom(lowerbound,upperbound));
    		 
    	 } 
    		 
    	}
     }
     

	public double sigmoid(double x) {
    	 return 0;
     }

     public double customSigmoid(double x) {
    	 return 0;
     }
     


     public void zeroWeigths() {
    	 for(ArrayList<Neuron> al: allLayerArrayList) {
    		 for(Neuron neuron: al) {
			ArrayList <Edge> inEdges = neuron.getInEdges();
			for(Edge e: inEdges) {
				e.setWeight(0);
			}
		}
      }
     }
     
     public double[] outputFor(double [] x) {
 		//setInputData(X);
    	// System.out.println(Arrays.deepToString(X));
		 //System.out.println(Arrays.toString(x));
    	 for(int i=0; i< inputLayer.size(); i++) {
    		 inputLayer.get(i).setNeuronOut(x[i+1]);
    	 }
    	 forwardPropagate();
 		 double  outputs[] = getOutputs();
 		 return outputs;
     }
     
     public double[] getOutputs() {
 		double [] outputs = new double[outputLayer.size()];
 		//System.out.println(outputLayer.size());
 		for(int i = 0; i < outputLayer.size(); i++) {
 			outputs[i] =outputLayer.get(i).getNeuronout();
 		}
 		return outputs;
     }
     
     public void forwardPropagate() {
    	for(ArrayList<Neuron> al: allLayerArrayList) {
    		for(Neuron n: al) {
			n.forwardPropagate();
		}
        }
     }
     
     public void backwardPropagate(double output[]) {
    	//int i = 0;              //?
 		for(Neuron n : outputLayer) {
 			double y = n.getNeuronout();
 			double z = output[0];
 			ArrayList<Edge> edges = n.getInEdges();
 			
 			for(Edge e : edges) {
 				double x = e.getInputValue();
 				double error = customSigmoidDerivative(y)*(z-y);
 				e.setError(error);
 				double delta =argMomentumTerm*e.getDelta() + argLearningRate*error*x; //current link's deltaweight has not be updated yet, so it is previous delta w
 				double newWeight = e.getWeight() + delta;
 				
 				e.setDelta(delta);
 				e.setWeight(newWeight);			
 			}
 			//i++;
 		}
 		//System.out.println("hey");
 		for(Neuron n: hiddenLayer) {   //different way to calculate error for nodes in hidden layer
 			double y =n.getNeuronout();
 			ArrayList<Edge> edges = n.getInEdges();
 			//System.out.println(edges.size());
 			for(Edge e : edges) {
 				double x = e.getInputValue();
 				double sumWeightedError = 0;
 				for(Neuron outNeuron: outputLayer) {
 					//System.out.println(edges.size());
 					double whj = outNeuron.getInEdgeMap(n.getNeuronId()).getWeight();
 					
 					double errorh = outNeuron.getInEdgeMap(n.getNeuronId()).getError();
 					sumWeightedError = sumWeightedError + whj *errorh;
 				}
 				double error = customSigmoidDerivative(y)*sumWeightedError;
 				e.setError(error);
 				double delta =argMomentumTerm * e.getDelta() + argLearningRate*error*x;
 				double newWeight = e.getWeight() + delta;
 				e.setDelta(delta);
 				e.setWeight(newWeight);							
 			}
 		}		
     }
     
     public double train(double[][] X, double[][] Y){  //one epoch
    	 double totalError = 0;
    	 for(int i=0; i<X.length; i++) {
    		 double error = 0;
    		 double outputZ[] = outputFor(X[i]);
    		// System.out.println(Arrays.deepToString(X));
    		// System.out.println(Arrays.toString(outputZ));
    		// System.out.println(outputZ[0]);
    		 for(int j = 0; j<argNumOutputs; j++) {
    			 error = error + Math.pow(outputZ[j]-Y[i][j], 2);
    		 }
    		 this.backwardPropagate(Y[i]);
    		 totalError = totalError + error;
    		 
    	 }
    	  totalErrorInEachEpoch.add(totalError);

    	 return totalError;
     }
     
     public double train(double [] x, double argValue) {
    	 return 0;
     }
     
     public void runNeuralNet(double errorThreshold,double[][] X, double[][] Y) {
    	int step = 1;
 		double error;
 		error = train(X,Y);
 		//System.out.print(error);
 		while(error > errorThreshold) {
 			error = train(X,Y);
 			step++;
 			totalEpochNum++;
 		}
 		System.out.println("Total error in the last epoch is " + error + "\n");
 		System.out.println("Total number of epoches "+ totalEpochNum + "\n");
     }
     
 	public ArrayList<Double> getErrorArray(){
		return this.totalErrorInEachEpoch;
	}
 	
     public void save(File argFile) {
    	 
     }
     
     public void load(String argFileName) throws IOException{
    	 
     }
     
     public double customSigmoidDerivative(double y) {
    	 double result;
    	 if(argA==-1) {
    		 result=1.0/2.0 * (1-y) * (1+y);
    	 }
    	 else {
    		 result=y*(1-y);
    	 }
    	 return result;
     }
     
  	public void printRunResults(ArrayList<Double> errors, String fileName) throws IOException {
 		int epoch;
 		PrintWriter printWriter = new PrintWriter(new FileWriter(fileName));
 		printWriter.printf("Epoch Number, Total Squared Error, \n");
 		for(epoch = 0; epoch < errors.size(); epoch++) {
 			printWriter.printf("%d, %f, \n", epoch, errors.get(epoch));
 		}
 		System.out.print("success!");
 		printWriter.flush();
 		printWriter.close();
 	}
}

