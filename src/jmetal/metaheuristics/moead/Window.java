package jmetal.metaheuristics.moead;

import java.io.Serializable;

/**
 * Class for "each slice of sliding window"
 * @author Jerry
 */
public class Window implements Serializable {
	
	private int index_;
	
	private double fitnessImprovement_;
	
	private int rank_;
	
	public Window() {
		index_              = 0;
		rank_               = 0;
		fitnessImprovement_ = 0;
	}
	
	/**
	 * Set the "index" of the current operator
	 * @param index
	 */
	public void setIndex(int index) {
		this.index_ = index;
	}
	
	/**
	 * Set the "fitness" value achieved by the application of a operator
	 * @param fitnessImprovement
	 */
	public void setFitness(double fitnessImprovement) {
		this.fitnessImprovement_ = fitnessImprovement;
	}
	
	/**
	 * Set the "rank" value of the application of a operator
	 * @param rank
	 */
	public void setRank(int rank) {
		this.rank_ = rank;
	}
	
	/**
	 * Get the "index" of the current operator
	 * @param index
	 */
	public int getIndex() {
		return this.index_;
	}
	
	/**
	 * Get the "fitness" value achieved by the application of a operator
	 * @param fitnessImprovement
	 */
	public double getFitness() {
		return this.fitnessImprovement_;
	}
	
	/**
	 * Get the "rank" value of the application of a operator
	 * @param rank
	 */
	public int getRank() {
		return this.rank_;
	}
}
