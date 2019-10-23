package Assignment1;

import java.util.ArrayList;

public class Edge {
	private double weight = 0;
	private Neuron pre;
	private Neuron next;
	private double inputValue = 0;
	private double error = 0;
	private double delta = 0; //
	
	 public Edge(Neuron pre, Neuron next) {
		 this.pre = pre;
		 this.next = next;
	 }
     public void setWeight(double weight) {
    	 this.weight = weight;
     }
     
     public void setDelta(double delta) {
    	 this.delta = delta;
     }
     
     public double getWeight() {
    	 return this.weight;
     }
     
     public double getDelta() {
    	 return this.delta;
     }
     
     public double getError() {
    	 return this.error;
     }
     
     public Neuron getPre() {
    	 return this.pre;
     }
     
     public Neuron getNext() {
    	 return this.next;
     }
     
     public double getInputValue() {
    	 inputValue = pre.getNeuronout();
    	 return inputValue;
     }
     
     public void setError(double error) {
    	 this.error = error;
     }
}
