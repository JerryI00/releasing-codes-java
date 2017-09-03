/**
 * MOEAD.java
 * 
 * This is the main implementation of MOEA/D-DE.
 * 
 * Reference:
 * 		Hui Li, Qingfu Zhang, Multiobjective Optimization Problems with 
 * 		Complicated Pareto Sets, MOEA/D and NSGA-II, IEEE Transactions on 
 * 		Evolutionary Computation, vol. 12, no 2, pp. 284-302, April, 2009
 */

package jmetal.metaheuristics.moead;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

import jmetal.util.*;

import java.util.Vector;

import jmetal.core.*;
import jmetal.util.PseudoRandom;

public class MOEAD extends Algorithm {
	
	private int         populationSize_;
	private SolutionSet population_;  // Population repository
	
	double[]   z_;					  // Z vector (ideal point)
	int[][]    neighborhood_; 		  // Neighborhood matrix
	double[][] lambda_; 			  // Lambda vectors
	
	int    T_;     					  // Neighborhood size
	int    nr_;    					  // Maximal number of solutions replaced by each child solution
	double delta_; 					  // Probability that parent solutions are selected from neighborhood
	int    evaluations_; 			  // Counter for the number of function evaluations
	
	Solution[] indArray_;
	String functionType_;
	
	Operator crossover_;
	Operator mutation_;

	String dataDirectory_;
	
	/**
	 * Constructor
	 * 
	 * @param Problem to solve
	 */
	public MOEAD(Problem problem) {
		super(problem);

		functionType_ = "_TCHE2";
//		functionType_ = "_PBI";
	} // DMOEA

	public SolutionSet execute() throws JMException, ClassNotFoundException {
		
		int maxEvaluations;

		evaluations_ = 0;
		maxEvaluations = ((Integer) this.getInputParameter("maxEvaluations")).intValue();
		populationSize_ = ((Integer) this.getInputParameter("populationSize")).intValue();
		dataDirectory_ = this.getInputParameter("dataDirectory").toString();
		System.out.println("POPSIZE: " + populationSize_);

		population_ = new SolutionSet(populationSize_);
		indArray_ = new Solution[problem_.getNumberOfObjectives()];

		T_ = 20;
		delta_ = 0.9;
		nr_ = 2;
				
		neighborhood_ = new int[populationSize_][T_];

		z_ = new double[problem_.getNumberOfObjectives()];
		lambda_ = new double[populationSize_][problem_.getNumberOfObjectives()];

		crossover_ = operators_.get("crossover"); 	// default: DE crossover
		mutation_ = operators_.get("mutation"); 	// default: polynomial mutation

		// STEP 1. Initialization
		// STEP 1.1. Compute Euclidean distances between weight vectors and find T
		initUniformWeight();
		initNeighborhood();

		// STEP 1.2. Initialize population
		initPopulation();

		// STEP 1.3. Initialize z_
		initIdealPoint();
		
		int idx = 0;
		String str1 = "FUN";
		String str2 = str1 + Integer.toString(idx);
		population_.printObjectivesToFile(str2);
					
		// STEP 2. Update
		do {		
			int[] permutation = new int[populationSize_];
			Utils.randomPermutation(permutation, populationSize_);

			for (int i = 0; i < populationSize_; i++) {
				int n = permutation[i];
				int type;
				double rnd = PseudoRandom.randDouble();

				// STEP 2.1. Mating selection based on probability
				if (rnd < delta_) // if (rnd < realb)
				{
					type = 1; // neighborhood
				} else {
					type = 2; // whole population
				}
				Vector<Integer> p = new Vector<Integer>();
				matingSelection(p, n, 2, type);

				// STEP 2.2. Reproduction
				Solution child;
				Solution[] parents = new Solution[3];

				parents[0] = population_.get(p.get(0));
				parents[1] = population_.get(p.get(1));
				parents[2] = population_.get(n);

				// Apply DE crossover
				child = (Solution) crossover_.execute(new Object[] {
						population_.get(n), parents });

				// Apply mutation
				mutation_.execute(child);

				// Evaluation
				problem_.evaluate(child);

				evaluations_++;

				// STEP 2.3. Repair. Not necessary

				// STEP 2.4. Update z_
				updateReference(child);

				// STEP 2.5. Update of solutions
				updateProblem(child, n, type);
			} // for
//			idx = medianPrint(idx);
			System.out.println(evaluations_);
		} while (evaluations_ < maxEvaluations);

		return population_;
	}
	
	/**
	 * print the median result
	 * @param idx
	 */
	public int medianPrint(int idx) {
		if (evaluations_ % 25000 == 0) {
			idx++;
			String str1 = "FUN";
			String str2 = str1 + Integer.toString(idx);
			
			population_.printObjectivesToFile(str2);
		}
		
		return idx;
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
	 * Update the current ideal point
	 * 
	 * @param individual
	 */
	void updateReference(Solution individual) {
		for (int n = 0; n < problem_.getNumberOfObjectives(); n++) {
			if (individual.getObjective(n) < z_[n]) {
				z_[n] = individual.getObjective(n);

				indArray_[n] = individual;
			}
		}
	} // updateReference

	/**
	 * Update the population by the current offspring
	 * 
	 * @param indiv: current offspring
	 * @param id:    index of current subproblem
	 * @param type:  update solutions in - neighborhood (1) or whole population (otherwise)
	 */
	void updateProblem(Solution indiv, int id, int type) {
		int size;
		int time;

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
				k = perm[i];
			}
			double f1, f2;

			f1 = fitnessFunction(population_.get(k), lambda_[k]);
			f2 = fitnessFunction(indiv, lambda_[k]);

			if (f2 < f1) {
				population_.replace(k, new Solution(indiv));
				time++;
			}
			// the maximal number of solutions updated is not allowed to exceed
			// 'limit'
			if (time >= nr_) {
				return;
			}
		}
	} // updateProblem

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
			double[] scale = new double[problem_.getNumberOfObjectives()];
			for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
				double min = 1.0e+30, max = -1.0e+30;
				for (int j = 0; j < population_.size(); j++) {
					Solution individal = population_.get(j);
					double tp = individal.getObjective(i);
					if (tp > max)
						max = tp;
					if (tp < min)
						min = tp;
				}
				scale[i] = max - min; // ���ÿһά��׼
				if (max - min == 0)
					return 1.0e+30;
			}

			double max_fun = -1.0e+30;
			for (int n = 0; n < problem_.getNumberOfObjectives(); n++) {
				double diff = (individual.getObjective(n) - z_[n]) / scale[n];
				double feval;
				if (lambda[n] == 0)
					feval = 0.0001 * diff;
				else
					feval = diff * lambda[n];
				if (feval > max_fun)
					max_fun = feval;

			}
			fitness = max_fun;
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

			// difference beween current point and reference point
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
} // MOEAD
