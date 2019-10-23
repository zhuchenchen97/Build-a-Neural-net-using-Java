package Assignment1;

import java.io.PipedInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Neuron {
	private String neuronId;
	private String activationFunction;
	private ArrayList <Edge> inEdges = new ArrayList <Edge>();
	private HashMap<String, Edge> allInEdges = new HashMap<String,Edge>();
	public double NeuronOut = 0; //neuron's value
	private double a;
	private double b;
    private Edge biasEdge;
	final double bias = 1;
	

	
	//Constuctor for input layer neurons
	public Neuron(String id,double a,double b) {
		this.neuronId = id;
		this.a = a;
		this.b = b;
	}
	// Constructor for hidden,output layer neurons
	public Neuron(String id, String activationFunction, List<Neuron> inNeurons, Neuron bias,double a,double b) {
		this.neuronId = id;
		this.activationFunction = activationFunction;
		this.a = a;
		this.b = b;
//		setActivationFunction(activationFunction);
		addInputEdges(inNeurons);
		addBiasInput(bias);
	}
    public Edge getBiasEdge() {
    	return this.biasEdge;
    }
    
	public double getNeuronout() {
		return this.NeuronOut;
	}
	
	public void setNeuronOut(double out) {
		this.NeuronOut = out;
	}
	
	public String getNeuronId() {
		return this.neuronId;
	}	
	
	public String getActivationFunction() {
		return this.activationFunction;
	}
	
    public ArrayList<Edge> getInEdges(){
    	 return this.inEdges;
     }
    
	public Edge getInEdgeMap(String neuronId) {
		return allInEdges.get(neuronId);
	}
    
//    public void setActivationFunction(String activationFunction) {
//    	
//    }
    
    public void addInputEdges(List<Neuron> inNeurons) {
    	for(Neuron neuron: inNeurons) {
    		Edge edge = new Edge(neuron,this);
    		inEdges.add(edge);
    		allInEdges.put(neuron.getNeuronId(), edge);
    	}
    }
    public void addBiasInput(Neuron bias) {
    	Edge edge = new Edge(bias, this);
    	inEdges.add(edge);
    	this.biasEdge = edge;
    	allInEdges.put(bias.getNeuronId(), edge);
    }
    
    public void forwardPropagate() {
    	double weightedSum = calculateWeightedSum(inEdges);
    	this.NeuronOut = customSigmoid(weightedSum);
    }
    
    public double calculateWeightedSum(ArrayList<Edge> inEdges) {
    	double sum = 0;
    	for (Edge e: inEdges){
    		double weight = e.getWeight();
    		double value = e.getInputValue();
			sum = sum + weight*value;
    	}
    	
    	if (biasEdge != null) {
			sum = sum + (this.biasEdge.getWeight()*this.bias);
    	}
		return sum;
    }

    public double sigmoid(double weightedSum) {
    	return 2/(1 + Math.exp(-weightedSum))-1;
    }
    public double customSigmoid(double weightedSum) {
    	return (b-a)/(1+Math.exp(-weightedSum))+a;
    }
    
}
