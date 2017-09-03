/**
 * MOEADDRA_MAB.java
 * 
 * This is main implementation of MOEA/D-FRRMAB
 * 
 * Author:
 * 		Ke Li <k.li@exeter.ac.uk>
 * 
 * Affliation:
 * 		Department of Computer Science, University of Exeter
 * 
 * Reference:
 * 		K. Li, A. Fialho, S. Kwong, Q. Zhang, 
 * 		"Adaptive Operator Selection with Bandits for Decomposition based Multi-Objective Optimization",
 * 		IEEE Transactions on Evolutionary Computation (TEVC), 18(1): 114¨C130, 2014. 
 * 
 * Homepage:
 * 		https://coda-group.github.io/
 * 
 * Copyright (c) 2017 Ke Li
 * 
 * Note: This is a free software developed based on the open source project 
 * jMetal<http://jmetal.sourceforge.net>. The copy right of jMetal belongs to 
 * its original authors, Antonio J. Nebro and Juan J. Durillo. Nevertheless, 
 * this current version can be redistributed and/or modified under the terms of 
 * the GNU Lesser General Public License as published by the Free Software 
 * Foundation, either version 3 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */
 
package jmetal.metaheuristics.moead;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import jmetal.util.*;

import java.util.Vector;

import jmetal.core.*;
import jmetal.util.PseudoRandom;
import jmetal.util.wrapper.XReal;

public class MOEADDRA_MAB extends Algorithm {

	private int         populationSize_;
	private SolutionSet population_;  // Population repository
	private Solution[]  savedValues_; // Individual repository
	
	double[][] lambda_;       		  // Lambda vectors
	int[][]    neighborhood_; 		  // Neighborhood matrix

	double[] z_; 	  				  // Z vector (ideal point)
	double[] objMax_; 			      // Maximum objective value
	
	int    T_;     					  // Neighborhood size
	int    nr_;    					  // Maximal number of solutions replaced by each child solution
	double delta_; 					  // Probability that parent solutions are selected from neighborhood
	
	int matingSize_; 				  // Maximal number of solutions to be selected as the parents
	int evaluations_; 				  // Counter for the number of function evaluations
	
	private int[]    frequency_;
	private double[] utility_;
	
	String     functionType_;
	Solution[] indArray_;
	
	String 	 dataDirectory_;
	Operator crossover_;
	Operator mutation_;
	
	/* AOS related parameters */
	int curSize_;					  // Current window size
	int windowSize_;				  // Length of the sliding window
	int numStrategies_; 			  // Number of operators
	
	int[] rank_;					  // Rank values of operators
	int[] strategyUsgae_;			  // Number of usages of operators
	int[] strategySelected_;		  // Indexes of the selected operators
	
	double scale_;					  // Scale factor to control the Exploration vs. Exploitation in FRRMAB model
	double decayFactor_;			  // Decay factor
	
	SlidingWindow slideWin_;
	
	double[] reward_;				  // Rewards of the applied operators
	double[] quality_;				  // Quality of the applied operators
	double[] decayReward_;			  // Decayed reward values
	double[] probability_;			  // Selection probability
	double[] improvement_;			  // Fitness improvement of each offspring
	double[][] slidingWindow_;		  // Sliding window

	/**
	 * Constructor
	 * 
	 * @param problem: Problem to solve
	 */
	public MOEADDRA_MAB(Problem problem) {
		super(problem);

//		functionType_ = "_TCHE1";
		functionType_ = "_TCHE2";
	} // constructor

	public SolutionSet execute() throws JMException, ClassNotFoundException {
		
		int maxEvaluations;
		
		int cur_id;
		int strategy_flag;

		int flag1, flag2, flag3, flag4;
		int uniform_flag, pre_flag, latter_flag;

		flag1 = flag2 = flag3 = flag4 = -1;
		uniform_flag = pre_flag = latter_flag = -1;

		evaluations_ = 0;
		
		maxEvaluations = ((Integer) this.getInputParameter("maxEvaluations"))
				.intValue();
		populationSize_ = ((Integer) this.getInputParameter("populationSize"))
				.intValue();
		dataDirectory_ = this.getInputParameter("dataDirectory").toString();

		population_  = new SolutionSet(populationSize_);
		savedValues_ = new Solution[populationSize_];
		utility_     = new double[populationSize_];
		frequency_   = new int[populationSize_];
		for (int i = 0; i < utility_.length; i++) {
			utility_[i] = 1.0;
			frequency_[i] = 0;
		}
		indArray_ = new Solution[problem_.getNumberOfObjectives()];

		T_     = 20;
		nr_    = 2;
		delta_ = 0.9;

		scale_       = 5.0;
		decayFactor_ = 1.0;
		windowSize_  = (int) (0.5 * populationSize_);
		curSize_     = 0;
				
		neighborhood_ = new int[populationSize_][T_];
		z_            = new double[problem_.getNumberOfObjectives()];
		objMax_       = new double[problem_.getNumberOfObjectives()];
		lambda_       = new double[populationSize_][problem_.getNumberOfObjectives()];

		/* AOS parameters initialization */
		numStrategies_ = 4;

		strategyUsgae_    = new int[numStrategies_];
		strategySelected_ = new int[populationSize_];
		reward_           = new double[numStrategies_];
		quality_          = new double[numStrategies_];
		improvement_      = new double[populationSize_];
		probability_      = new double[numStrategies_];

		rank_          = new int[numStrategies_];
		decayReward_   = new double[numStrategies_];
		slidingWindow_ = new double[2][windowSize_];
		slideWin_      = new SlidingWindow(windowSize_);

		for (int i = 0; i < numStrategies_; i++) {
			rank_[i]        = 0;
			decayReward_[i] = 0.0;
		}
		for (int i = 0; i < windowSize_; i++) {
			slidingWindow_[0][i] = -1;
			slidingWindow_[1][i] = -1;
		}

		/* Initialize the Adaptive system related arrays */
		initParameter(probability_, quality_, reward_);
		
		/* Default: DE crossover */
		crossover_ = operators_.get("crossover");
		
		/* Default: Polynomial mutation */
		mutation_ = operators_.get("mutation");

		/** 
		 * STEP 1: Initialization 
		 */		
		/* STEP 1.1: Compute Euclidean distances between weight vectors and find T */
		initUniformWeight();
		initNeighborhood();

		/* STEP 1.2: Initialize population */
		initPopulation();

		/* STEP 1.3: Initialize z_ */
		initIdealPoint();

		int gen = 0;
		
		/** 
		 * STEP 2: Update Procedure
		 */
		do {
			for (int i = 0; i < populationSize_; i++) {
				improvement_[i] = 0.0;
			}

			int[] permutation = new int[populationSize_];
			Utils.randomPermutation(permutation, populationSize_);
			List<Integer> order = tour_selection(10);

			for (int i = 0; i < order.size(); i++) {
				cur_id = order.get(i);
				frequency_[cur_id]++;

				if (uniform_flag == -1) {
					strategy_flag = (int) Math.ceil(PseudoRandom.randDouble(
							0.0, 1.0) * numStrategies_);
					switch (strategy_flag) {
					case 1:
						flag1 = 1;
						break;
					case 2:
						flag2 = 1;
						break;
					case 3:
						flag3 = 1;
						break;
					case 4:
						flag4 = 1;
						break;
					}

					if (flag1 == 1 && flag2 == 1) {
						pre_flag = 1;
					}
					if (flag3 == 1 && flag4 == 1) {
						latter_flag = 1;
					}
					if (pre_flag == 1 && latter_flag == 1) {
						uniform_flag = 1;
					}
				} else {
					strategy_flag = AdaptiveModule.FRRMAB(quality_,
							decayReward_, strategyUsgae_, numStrategies_,
							scale_);
				}
				strategySelected_[i] = strategy_flag;

				matingEvolution(strategy_flag, cur_id);

				Refreshment(strategyUsgae_, reward_, decayReward_,
						numStrategies_);

				Window current_element;

				current_element = new Window();
				current_element.setIndex(strategy_flag);
				current_element.setFitness(improvement_[cur_id]);

				if (curSize_ < windowSize_) {
					slideWin_.add(current_element);
					curSize_++;
				} else {
					slideWin_.remove(0);
					slideWin_.add(current_element);
				}
				slideWin_.readSlidingWindow(slidingWindow_);

				AdaptiveModule.update_rewards(slidingWindow_, reward_,
						strategyUsgae_, curSize_);
				AdaptiveModule.Rank_rewards(reward_, numStrategies_, rank_);
				AdaptiveModule.CreaditAssignmentDecay(reward_, decayReward_,
						numStrategies_, rank_, decayFactor_);
			} // for

			gen++;
			if (gen % 50 == 0) {
				comp_utility();
			}
		} while (evaluations_ < maxEvaluations);

		return population_;
	}

	/**
	 * Initialize the weight vectors, this function only can read from the 
	 * existing data file, instead of generating itself.
	 * 
	 */
	public void initUniformWeight() {
		String dataFileName;
		dataFileName = "W" + problem_.getNumberOfObjectives() + "D_"
				+ populationSize_ + ".dat";

		try {
			// Open the file
			FileInputStream fis = new FileInputStream(dataDirectory_ + "/"
					+ dataFileName);
			InputStreamReader isr = new InputStreamReader(fis);
			BufferedReader br = new BufferedReader(isr);

			int i = 0;
			int j = 0;
			String aux = br.readLine();
			while (aux != null) {
				StringTokenizer st = new StringTokenizer(aux);
				j = 0;
				while (st.hasMoreTokens()) {
					double value = (new Double(st.nextToken())).doubleValue();
					lambda_[i][j] = value;
					j++;
				}
				aux = br.readLine();
				i++;
			}
			br.close();
		} catch (Exception e) {
			System.out
					.println("initUniformWeight: failed when reading for file: "
							+ dataDirectory_ + "/" + dataFileName);
			e.printStackTrace();
		}
	} // initUniformWeight

	/**
	 * Update the utilities of subproblems
	 * 
	 */
	public void comp_utility() {
		double f1, f2, uti, delta;

		for (int i = 0; i < populationSize_; i++) {
			f1 = fitnessFunction(population_.get(i), lambda_[i]);
			f2 = fitnessFunction(savedValues_[i], lambda_[i]);

			delta = (f2 - f1) / f2;
			if (delta > 0.001)
				utility_[i] = 1.0;
			else {
				uti = 0.95 * (1.0 + delta / 0.001) * utility_[i];
				utility_[i] = uti < 1.0 ? uti : 1.0;
			}
			savedValues_[i] = new Solution(population_.get(i));
		}
	}

	/**
	 * Initialize the neighborhood matrix of subproblems, based on the Euclidean
	 * distances between different weight vectors
	 * 
	 */
	public void initNeighborhood() {
		int[] idx  = new int[populationSize_];
		double[] x = new double[populationSize_];

		for (int i = 0; i < populationSize_; i++) {
			/* calculate the distances based on weight vectors */
			for (int j = 0; j < populationSize_; j++) {
				x[j] = Utils.distVector(lambda_[i], lambda_[j]);
				idx[j] = j;
			}
			/* find 'niche' nearest neighboring subproblems */
			Utils.minFastSort(x, idx, populationSize_, T_);

			for (int k = 0; k < T_; k++) {
				neighborhood_[i][k] = idx[k];
			}
		}
	} // initNeighborhood

	/**
	 * Initialize the population, random sampling from the search space
	 * 
	 * @throws JMException
	 * @throws ClassNotFoundException
	 */
	public void initPopulation() throws JMException, ClassNotFoundException {
		for (int i = 0; i < populationSize_; i++) {
			Solution newSolution = new Solution(problem_);

			problem_.evaluate(newSolution);
			evaluations_++;
			population_.add(newSolution);
			savedValues_[i] = new Solution(newSolution);
		}
	} // initPopulation

	/**
	 * Initialize the ideal point, the best objective function value for each
	 * individual objective
	 * 
	 * @throws JMException
	 * @throws ClassNotFoundException
	 */
	void initIdealPoint() throws JMException, ClassNotFoundException {
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			z_[i] = 1.0e+30;

		for (int i = 0; i < populationSize_; i++)
			updateReference(population_.get(i));
	} // initIdealPoint

	/**
	 * Select the mating parents, depending on the selection 'type'
	 * 
	 * @param list : the set of the indexes of selected mating parents
	 * @param cid  : the id of current subproblem
	 * @param size : the number of selected mating parents
	 * @param type : 1 - neighborhood; otherwise - whole population
	 */
	public void matingSelection(Vector<Integer> list, int cid, int size, int type) {
		int ss;
		int r;
		int p;

		ss = neighborhood_[cid].length;
		while (list.size() < size) {
			if (type == 1) {
				r = PseudoRandom.randInt(0, ss - 1);
				p = neighborhood_[cid][r];
			} else {
				p = PseudoRandom.randInt(0, populationSize_ - 1);
			}
			boolean flag = true;
			for (int i = 0; i < list.size(); i++) {
				if (list.get(i) == p) // p is in the list
				{
					flag = false;
					break;
				}
			}

			if (flag) {
				list.addElement(p);
			}
		}
	} // matingSelection

	/**
	 * Tournament selection
	 * 
	 * @param depth: tournament size
	 * @return
	 */
	public List<Integer> tour_selection(int depth) {

		// selection based on utility
		List<Integer> selected = new ArrayList<Integer>();
		List<Integer> candidate = new ArrayList<Integer>();

		for (int k = 0; k < problem_.getNumberOfObjectives(); k++)
			selected.add(k); // select first m weights
		for (int n = problem_.getNumberOfObjectives(); n < populationSize_; n++)
			candidate.add(n); // set of unselected weights

		while (selected.size() < (int) (populationSize_ / 5.0)) {
			int best_idd = (int) (PseudoRandom.randDouble() * candidate.size());
			int i2;
			int best_sub = candidate.get(best_idd);
			int s2;
			for (int i = 1; i < depth; i++) {
				i2 = (int) (PseudoRandom.randDouble() * candidate.size());
				s2 = candidate.get(i2);
				
				if (utility_[s2] > utility_[best_sub]) {
					best_idd = i2;
					best_sub = s2;
				}
			}
			selected.add(best_sub);
			candidate.remove(best_idd);
		}
		return selected;
	}

	/**
	 * Update the current ideal point
	 * 
	 * @param individual
	 */
	void updateReference(Solution individual) {
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
			if (individual.getObjective(i) < z_[i])
				z_[i] = individual.getObjective(i);
		}
	} // updateReference

	double innerproduct(double[] vec1, double[] vec2) {
		double sum = 0;
		for (int i = 0; i < vec1.length; i++)
			sum += vec1[i] * vec2[i];
		return sum;
	}

	double norm_vector(Vector<Double> x) {
		double sum = 0.0;
		for (int i = 0; i < (int) x.size(); i++)
			sum = sum + x.get(i) * x.get(i);
		return Math.sqrt(sum);
	}

	double norm_vector(double[] z) {
		double sum = 0.0;
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			sum = sum + z[i] * z[i];
		return Math.sqrt(sum);
	}
	
	/**
	 * Evaluate the fitness function by the decomposition method
	 * 
	 * @param individual: current solution
	 * @param lambda:     weight vector
	 * @return
	 */
	double fitnessFunction(Solution individual, double[] lambda) {
		double fitness;
		fitness = 0.0;

		if (functionType_.equals("_TCHE1")) {
			double maxFun = -1.0e+30;

			for (int n = 0; n < problem_.getNumberOfObjectives(); n++) {
				double diff = Math.abs(individual.getObjective(n) - z_[n]);

				double feval;
				if (lambda[n] == 0) {
					feval = 0.0001 * diff;
				} else {
					feval = diff * lambda[n];
				}
				if (feval > maxFun) {
					maxFun = feval;
				}
			} // for

			fitness = maxFun;
		} else if (functionType_.equals("_TCHE2")) {
			double maxFun = -1.0e+30;

			for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
				double diff = Math.abs(individual.getObjective(i) - z_[i]);

				double feval;
				if (lambda[i] == 0) {
					feval = diff / 0.000001;
				} else {
					feval = diff / lambda[i];
				}
				if (feval > maxFun) {
					maxFun = feval;
				}
			} // for
			fitness = maxFun;
		} else if (functionType_.equals("_PBI"))// if
		{
			double theta; // penalty parameter
			theta = 5.0;

			// normalize the weight vector (line segment)
			double nd = norm_vector(lambda);
			for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
				lambda[i] = lambda[i] / nd;

			double[] realA = new double[problem_.getNumberOfObjectives()];
			double[] realB = new double[problem_.getNumberOfObjectives()];

			// difference between current point and reference point
			for (int n = 0; n < problem_.getNumberOfObjectives(); n++)
				realA[n] = (individual.getObjective(n) - z_[n]);

			// distance along the line segment
			double d1 = Math.abs(innerproduct(realA, lambda));

			// distance to the line segment
			for (int n = 0; n < problem_.getNumberOfObjectives(); n++)
				realB[n] = (individual.getObjective(n) - (z_[n] + d1
						* lambda[n]));
			double d2 = norm_vector(realB);

			fitness = d1 + theta * d2;
		} else {
			System.out.println("MOEAD.fitnessFunction: unknown type "
					+ functionType_);
			System.exit(-1);
		}
		return fitness;
	} // fitnessEvaluation
	
	/**
	 * Initialize the parameter of the AOS module
	 * 
	 * @param probability: selection probability of each operator
	 * @param quality:     quality of each operator
	 * @param reward:      reward values of each operator
	 */
	public void initParameter(double[] probability, double[] quality,
			double[] reward) {
		int i;

		for (i = 0; i < numStrategies_; i++) {
			probability[i] = 1.0 / (double) (numStrategies_);
			quality[i] = 0.0;
			reward[i] = 0.0;
			strategyUsgae_[i] = 0;
		}
	} // AOS parameters initialization

	/**
	 * Refresh the related ingredients of the FRRMAB model
	 * 
	 * @param strategyUsage: number of usages for each operator
	 * @param rewards:       reward values for each operator
	 * @param decayRewards:  decayed reward values
	 * @param numStrategies: number of operators
	 */
	public void Refreshment(int[] strategyUsage, double[] rewards,
			double[] decayRewards, int numStrategies) {
		int i;

		for (i = 0; i < numStrategies; i++) {
			strategyUsage[i] = 0;
			decayRewards[i] = 0.0;
			rewards[i] = 0.0;
		}
	}

	/**
	 * Main function for offspring generation
	 * 
	 * @param strategySelected: index of the selected operator
	 * @param cur_id:           index of the current subproblem
	 * @throws JMException
	 */
	public void matingEvolution(int strategySelected, int cur_id)
			throws JMException {
		int type;
		double rnd;
		
		Solution child = null;
		Solution[] parents;

		Vector<Integer> p = new Vector<Integer>();

		rnd = PseudoRandom.randDouble();
		/* STEP 2.1: Mating selection based on probability */
		if (rnd < delta_) {
			type = 1; // neighborhood
		} else {
			type = 2; // whole population
		}

		switch (strategySelected) {
		case 1:
			/**
			 * DE/rand/1/bin: u = x1 + F * (x2 - x3)
			 */
			matingSize_ = 3;
			matingSelection(p, cur_id, matingSize_, type);

			/* STEP 2.2: Reproduction */
			parents = new Solution[3];

			parents[0] = population_.get(p.get(0));
			parents[1] = population_.get(p.get(1));
			parents[2] = population_.get(p.get(2));

			crossover_.setParameter("DE_VARIANT", "rand/1/bin");
			/* Apply DE operator */
			child = (Solution) crossover_.execute(new Object[] {
					population_.get(cur_id), parents });

			/* Apply mutation */
			mutation_.execute(child);

			/* Function evaluation */
			problem_.evaluate(child);
			evaluations_++;

			/* STEP 2.3: Repair. Not necessary */

			/* STEP 2.4: Update ideal point z_ */
			updateReference(child);

			/* STEP 2.5: Update the current subproblem */
			updateProblemOrigin(child, cur_id, type);

			p.clear();
			break;
		case 2:
			/** 
			 * DE/rand/2/bin: u = xi + F * (x1 - x2) + F * (x3 - x4) 
			 */
			matingSize_ = 4;
			matingSelection(p, cur_id, matingSize_, type);

			/* STEP 2.2: Reproduction */
			parents = new Solution[5];

			parents[0] = population_.get(p.get(0));
			parents[1] = population_.get(p.get(1));
			parents[2] = population_.get(p.get(2));
			parents[3] = population_.get(p.get(3));
			parents[4] = population_.get(cur_id);

			crossover_.setParameter("DE_VARIANT", "DE/rand/2/bin");
			/* Apply DE operator */
			child = (Solution) crossover_.execute_5(new Object[] {
					population_.get(cur_id), parents });

			/* Apply mutation */
			mutation_.execute(child);

			/* Function evaluation */
			problem_.evaluate(child);
			evaluations_++;

			/* STEP 2.3: Repair. Not necessary */

			/* STEP 2.4: Update ideal point z_ */
			updateReference(child);

			/* STEP 2.5: Update the current subproblem */
			updateProblemOrigin(child, cur_id, type);

			p.clear();
			break;
		case 3:
			/**
			 * DE/current-to-rand/2/bin: u = xi + K * (x1 - xi) + F * (x2 - x3) + F * (x4 - x5)
			 */
			matingSize_ = 5;
			matingSelection(p, cur_id, matingSize_, type);

			/* STEP 2.2: Reproduction */
			parents = new Solution[6];

			parents[0] = population_.get(p.get(0));
			parents[1] = population_.get(p.get(1));
			parents[2] = population_.get(p.get(2));
			parents[3] = population_.get(p.get(3));
			parents[4] = population_.get(p.get(4));
			parents[5] = population_.get(cur_id);

			crossover_.setParameter("DE_VARIANT", "DE/current-to-rand/2/bin");
			/* Apply DE operator */
			child = (Solution) crossover_.execute_6(new Object[] {
					population_.get(cur_id), parents });

			/* Apply mutation */
			mutation_.execute(child);

			/* Function evaluation */
			problem_.evaluate(child);
			evaluations_++;

			/* STEP 2.3: Repair. Not necessary */

			/* STEP 2.4: Update ideal point z_ */
			updateReference(child);

			/* STEP 2.5: Update the current subproblem */
			updateProblemOrigin(child, cur_id, type);

			p.clear();
			break;
		case 4:
			/** 
			 * DE/current-to-rand/1/bin: u = xi + K * (xi - x1) + F * (x2 - x3)
			 */
			matingSize_ = 3;
			matingSelection(p, cur_id, matingSize_, type);

			/* STEP 2.2: Reproduction */
			parents = new Solution[3];
			parents[0] = population_.get(p.get(0));
			parents[1] = population_.get(p.get(1));
			parents[2] = population_.get(p.get(2));

			crossover_.setParameter("DE_VARIANT", "current-to-rand/1/bin");

			/* Apply DE operator */
			child = (Solution) crossover_.execute(new Object[] {
					population_.get(cur_id), parents });

			/* Apply mutation */
			mutation_.execute(child);

			/* Function evaluation */
			problem_.evaluate(child);

			evaluations_++;

			/* STEP 2.3: Repair. Not necessary */

			/* STEP 2.4: Update ideal point z_ */
			updateReference(child);

			/* STEP 2.5: Update the current subproblem */
			updateProblemOrigin(child, cur_id, type);

			p.clear();
			break;
		}
	}
	
	/**
	 * Update the population by the current offspring
	 * 
	 * @param indiv : offspring solution
	 * @param id    : the id of current subproblem
	 * @param type  : update solutions in - neighborhood (1) or whole population (otherwise)
	 * @throws JMException
	 */
	void updateProblemOrigin(Solution indiv, int id, int type)
			throws JMException {
		int size;
		int time;
		double fitnessimprovement;
		
		time = 0;

		if (type == 1) {
			size = neighborhood_[id].length;
		} else {
			size = population_.size();
		}
		int[] perm = new int[size];

		Utils.randomPermutation(perm, size);

		for (int i = 0; i < size; i++) {
			int k;
			if (type == 1) {
				k = neighborhood_[id][perm[i]];
			} else {
				/* Calculate the values of objective function regarding the current subproblem */
				k = perm[i];
			}
			double f1, f2;

			f1 = fitnessFunction(population_.get(k), lambda_[k]);
			f2 = fitnessFunction(indiv, lambda_[k]);
			
			fitnessimprovement = (f1 - f2) / f1;

			if (fitnessimprovement > 0) {
				population_.replace(k, new Solution(indiv));
				improvement_[id] += fitnessimprovement;
				time++;
			}
			/* Maximal number of solutions updated is not allowed to exceed 'limit' */
			if (time >= nr_) {
				return;
			}
		}
	} // updateProblemOrigin
	
} // MOEA/D-FRRMAB

