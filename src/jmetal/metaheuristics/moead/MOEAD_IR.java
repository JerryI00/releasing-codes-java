/**
 * MOEAD_IR.java
 * 
 * This is main implementation of MOEA/D-IR.
 * 
 * Author:
 * 		Ke Li <k.li@exeter.ac.uk>
 * 
 * Affliation:
 * 		Department of Computer Science, University of Exeter
 * 
 * Reference:
 * 		K. Li, S. Kwong, Q. Zhang, K. Deb, 
 * 		"Inter-Relationship Based Selection for Decomposition Multiobjective Optimization"
 * 		IEEE Transactions on Cybernetics (TCYB), 45(10): 2076¨C2088, 2015 
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
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.StringTokenizer;
import jmetal.util.*;

import java.util.Vector;

import jmetal.core.*;
import jmetal.util.PseudoRandom;

public class MOEAD_IR extends Algorithm {

	private int populationSize_;

	private SolutionSet population_;
	private SolutionSet currentOffspring_;
	private SolutionSet union_;
	
	private Solution[] savedValues_;

	private double[] utility_;

	double[] z_;  			// ideal point
	double[] nz_; 			// nadir point

	double[][] lambda_; 	// weight vectors
	int[][] neighborhood_;  // neighborhood structure
	
	int T_; 				// neighborhood size
	double delta_;  		// probability that parent solutions are selected from neighborhood	
	
	int Kd_; 				// the maximum number subproblems should be related to a solution
	int theta_;  			// the maximum number of solutions should be related to a subproblem

	int evaluations_;
	
	String functionType_;
	
	Operator crossover_;
	Operator mutation_;

	String dataDirectory_;

  	/**
  	 * Constructor
  	 * @param problem Problem to solve
  	 */
	public MOEAD_IR(Problem problem) {
		super(problem);

		// functionType_ = "_TCHE1";
		functionType_ = "_TCHE2";
		// functionType_ = "_PBI";

	} // MOEA/D-D

	public SolutionSet execute() throws JMException, ClassNotFoundException {
		int maxEvaluations;

		evaluations_    = 0;
		maxEvaluations  = ((Integer) this.getInputParameter("maxEvaluations")).intValue();
		populationSize_ = ((Integer) this.getInputParameter("populationSize")).intValue();
		dataDirectory_  = this.getInputParameter("dataDirectory").toString();

		population_  = new SolutionSet(populationSize_);
		savedValues_ = new Solution[populationSize_];
		utility_     = new double[populationSize_];
		for (int i = 0; i < utility_.length; i++)
			utility_[i] = 1.0;

		T_     = 20;
		delta_ = 0.9;
		Kd_    = 2;
		theta_ = 8;

		z_ 			  = new double[problem_.getNumberOfObjectives()];
	    nz_ 		  = new double[problem_.getNumberOfObjectives()];
	    lambda_ 	  = new double[populationSize_][problem_.getNumberOfObjectives()];
	    neighborhood_ = new int[populationSize_][T_];

		crossover_ = operators_.get("crossover"); // default: DE crossover
		mutation_  = operators_.get("mutation"); // default: polynomial mutation

		// STEP 1. Initialization
		// STEP 1.1. Compute Euclidean distances between weight vectors and find T
		initUniformWeight();
		initNeighborhood();

		// STEP 1.2. Initialize population
		initPopulation();

		// STEP 1.3. Initialize the ideal point 'z_' and the nadir point 'nz_'
		initIdealPoint();
		initNadirPoint();

		int iteration = 0;
		// STEP 2. Update
		do {
			// Select the satisfied subproblems
			List<Integer> order = tour_selection(10);

			currentOffspring_ = new SolutionSet(order.size());
			
			for (int i = 0; i < order.size(); i++) {
				int n = order.get(i);

				int type;
				double rnd = PseudoRandom.randDouble();

				// STEP 2.1. Mating selection based on probability
				if (rnd < delta_)
				{
					type = 1; // neighborhood
				} else {
					type = 2; // whole population
				}
				Solution child;
				Solution[] parents = new Solution[3];
				Vector<Integer> p = new Vector<Integer>();
				
				parents = matingSelection(p, n, 2, type);
				
				// Apply DE crossover and polynomial mutation
				child = (Solution) crossover_.execute(new Object[] {population_.get(n), parents});
				mutation_.execute(child);

				// Evaluation
				problem_.evaluate(child);
				evaluations_++;

				/* STEP 2.3. Update the ideal point 'z_' and nadir point 'nz_' */
				updateReference(child);
				
				// Add into the offspring population
				currentOffspring_.add(child);
			} // for
			
			// Combine the parent and the current offspring populations
			union_ = ((SolutionSet) population_).union(currentOffspring_);
			
			// Selection Procedure
			selection();

			// Update the utility value of subproblems
			iteration++;
			if (iteration % 30 == 0) {
				comp_utility();
			}
		} while (evaluations_ < maxEvaluations);
		
		return population_;
	}
		
	/**
  	 * Select the next parent population, based on the inter-relationships
  	 */
	public void selection() {
		
		int[] idx 			= new int[populationSize_];		// The indices of the solutions that have finally been selected for the parents
		int[] selected		= new int[union_.size()];       // If a solution 'i' is selected by a subproblem, selected[i] = 1, otherwise -1
		int[] emptySubp 	= new int[populationSize_];		// The record of subproblem that has not selected solutions
		double[] nicheCount = new double[populationSize_];

		int[][] solPref      	 = new int[union_.size()][];	// The indices of the subproblems that are preferred by solutions
		double[][] solMatrix 	 = new double[union_.size()][];	// The preference values of the subproblems on solutions
		double[][] distMatrix    = new double[union_.size()][];
		double[][] fitnessMatrix = new double[union_.size()][];
		for (int i = 0; i < union_.size(); i++) {
			solPref[i]   	 = new int[populationSize_];
			solMatrix[i] 	 = new double[populationSize_];
			distMatrix[i]    = new double[populationSize_];
			fitnessMatrix[i] = new double[populationSize_];
		}
		int[][] subpPref = new int[populationSize_][];	// The indices of the solutions that are preferred by subproblems
		for (int i = 0; i < populationSize_; i++) {
			subpPref[i] = new int[theta_];
		}

		// Initialize the niche count and idx
		for (int i = 0; i < populationSize_; i++) {
			idx[i] 	      = -1;
			nicheCount[i] = 0;
		}
		
		// Initialize the index of selected solutions to be -1
		for (int i = 0; i < union_.size(); i++) {
			selected[i] = -1;
		}
		
		// Initialize the neighborhood index of subproblems to be -1
		for (int i = 0; i < populationSize_; i++) {
			for (int j = 0; j < theta_; j++) {
				subpPref[i][j] = -1;
			}
		}
		
		// Calculate the preference values of solution matrix
		for (int i = 0; i < union_.size(); i++) {
			int minIndex = 0;
			for (int j = 0; j < populationSize_; j++) {
				fitnessMatrix[i][j] = fitnessFunction(union_.get(i), lambda_[j]);
				distMatrix[i][j]  	= calculateDistance2(union_.get(i), lambda_[j]);
			 	if (distMatrix[i][j] < distMatrix[i][minIndex])
			 		minIndex = j;
			}
			nicheCount[minIndex] = nicheCount[minIndex] + 1;
		}
		
		// Calculate the preference value of subproblems on solutions
		for (int i = 0; i < union_.size(); i++) {
			for (int j = 0; j < populationSize_; j++)
				solMatrix[i][j] = distMatrix[i][j] + nicheCount[j];
		}
		
		// Sort the preference value matrix of solution on subproblems
		for (int i = 0; i < union_.size(); i++) {
			for (int j = 0; j < populationSize_; j++)
				solPref[i][j] = j;
			Utils.QuickSort(solMatrix[i], solPref[i], 0, populationSize_ - 1);
		}
		
		for (int i = 0; i < populationSize_; i++) {
			int length = 0;
			for (int j = 0; j < union_.size(); j++) {
				if (length < theta_) {
					for (int k = 0; k < Kd_; k++) {
						if (length < theta_) {
							if (solPref[j][k] == i) {
								subpPref[i][length] = j;
								length++;
								break;
							}
						} else
							break;			
					}	
				} else
					break;
			}
		}
		
		// Making decision on the selection of solutions;
		int no_unselected = 0;
		for (int i = 0; i < populationSize_; i++) {
			if (subpPref[i][0] != -1) {
				int minIndex  = 0;
				double curMin = Double.MAX_VALUE;
				for (int j = 0; j < theta_; j++) {
					if (subpPref[i][j] != -1) {
						double tempfitness = fitnessMatrix[subpPref[i][j]][i];
						if (tempfitness < curMin) {
							curMin   = tempfitness;
							minIndex = subpPref[i][j];
						}
					} else {
						break;
					}
				}
				idx[i] 			 = minIndex;
				selected[idx[i]] = 1;
			} else {
				emptySubp[no_unselected] = i;
				no_unselected++;
			}
		}

		// Selection for the unselected subproblems
		if (no_unselected > 0) {
			for (int i = 0; i < no_unselected; i++) {
				int minIndex  = 0;
				double curMin = Double.MAX_VALUE;
				for (int j = 0; j < union_.size(); j++) {
					if (selected[j] == -1) {
						double tempfitness = fitnessMatrix[j][emptySubp[i]]; 
						if (tempfitness < curMin) {
							curMin 	 = tempfitness;
							minIndex = j;
						}
					}
				}
				idx[emptySubp[i]]  = minIndex;
				selected[minIndex] = 1;
			}
		}
		
		for (int i = 0; i < populationSize_; i++)
			population_.replace(i, new Solution(union_.get(idx[i])));
	}
	
	/**
  	 * Select the next parent population, based on the inter-relationships
  	 */
	public void selection_Backup() {
		
		int[] idx 			= new int[populationSize_];		// The indices of the solutions that have finally been selected for the parents
		int[] selected		= new int[union_.size()];       // If a solution 'i' is selected by a subproblem, selected[i] = 1, otherwise -1
		int[] emptySubp 	= new int[populationSize_];		// The record of subproblem that has not selected solutions
		double[] nicheCount = new double[populationSize_];

		int[][] solPref      	 = new int[union_.size()][];	// The indices of the subproblems that are preferred by solutions
		double[][] solMatrix 	 = new double[union_.size()][];	// The preference values of the subproblems on solutions
		double[][] distMatrix    = new double[union_.size()][];
		double[][] fitnessMatrix = new double[union_.size()][];
		for (int i = 0; i < union_.size(); i++) {
			solPref[i]   	 = new int[populationSize_];
			solMatrix[i] 	 = new double[populationSize_];
			distMatrix[i]    = new double[populationSize_];
			fitnessMatrix[i] = new double[populationSize_];
		}
		int[][] subpPref = new int[populationSize_][];	// The indices of the solutions that are preferred by subproblems
		for (int i = 0; i < populationSize_; i++) {
			subpPref[i] = new int[theta_];
		}

		// Initialize the niche count and idx
		for (int i = 0; i < populationSize_; i++) {
			idx[i] 	      = -1;
			nicheCount[i] = 0;
		}
		
		// Initialize the index of selected solutions to be -1
		for (int i = 0; i < union_.size(); i++) {
			selected[i] = -1;
		}
		
		// Initialize the neighborhood index of subproblems to be -1
		for (int i = 0; i < populationSize_; i++) {
			for (int j = 0; j < theta_; j++) {
				subpPref[i][j] = -1;
			}
		}
		
		// Calculate the preference values of solution matrix
		for (int i = 0; i < union_.size(); i++) {
			int minIndex = 0;
			for (int j = 0; j < populationSize_; j++) {
				fitnessMatrix[i][j] = fitnessFunction(union_.get(i), lambda_[j]);
				distMatrix[i][j]  	= calculateDistance(union_.get(i), lambda_[j]);
			 	if (distMatrix[i][j] < distMatrix[i][minIndex])
			 		minIndex = j;
			}
			nicheCount[minIndex] = nicheCount[minIndex] + 1;
		}
		
		// Calculate the preference value of subproblems on solutions
		for (int i = 0; i < union_.size(); i++) {
			for (int j = 0; j < populationSize_; j++)
				solMatrix[i][j] = fitnessMatrix[i][j] + nicheCount[j];
		}
		
		// Sort the preference value matrix of solution on subproblems
		for (int i = 0; i < union_.size(); i++) {
			for (int j = 0; j < populationSize_; j++)
				solPref[i][j] = j;
			Utils.QuickSort(solMatrix[i], solPref[i], 0, populationSize_ - 1);
		}
				
		for (int i = 0; i < populationSize_; i++) {
			int length = 0;
			for (int j = 0; j < union_.size(); j++) {
				if (length < theta_) {
					for (int k = 0; k < Kd_; k++) {
						if (length < theta_) {
							if (solPref[j][k] == i) {
								subpPref[i][length] = j;
								length++;
								break;
							}
						} else
							break;			
					}	
				} else
					break;
			}
		}
		
		// Making decision on the selection of solutions;
		int no_unselected = 0;
		for (int i = 0; i < populationSize_; i++) {
			if (subpPref[i][0] != -1) {
				int minIndex  = 0;
				double curMin = Double.MAX_VALUE;
				for (int j = 0; j < theta_; j++) {
					if (subpPref[i][j] != -1) {
						double tempfitness = fitnessFunction(union_.get(subpPref[i][j]), lambda_[i]);
						if (tempfitness < curMin) {
							curMin   = tempfitness;
							minIndex = subpPref[i][j];
						}
					} else {
						break;
					}
				}
				idx[i] 			 = minIndex;
				selected[idx[i]] = 1;
			} else {
				emptySubp[no_unselected] = i;
				no_unselected++;
			}
		}

		// Selection for the unselected subproblems
		if (no_unselected > 0) {
			for (int i = 0; i < no_unselected; i++) {
				int minIndex  = 0;
				double curMin = Double.MAX_VALUE;
				for (int j = 0; j < union_.size(); j++) {
					if (selected[j] == -1) {
						double tempfitness = fitnessFunction(union_.get(j), lambda_[emptySubp[i]]);
						if (tempfitness < curMin) {
							minIndex = j;
							curMin 	 = tempfitness;
						}
					}
				}
				idx[emptySubp[i]]  = minIndex;
				selected[minIndex] = 1;
			}
		}
		
		for (int i = 0; i < populationSize_; i++)
			population_.replace(i, new Solution(union_.get(idx[i])));
	}

	/**
	 * Calculate the perpendicular distance between the solution and reference line
	 * @param individual
	 * @param lambda
	 * @return
	 */
	public double calculateDistance(Solution individual, double[] lambda) {
		
		double scale;
		double distance;

		double[] vecInd  = new double[problem_.getNumberOfObjectives()];
		double[] vecProj = new double[problem_.getNumberOfObjectives()];
		
		// vecInd has been normalized to the range [0,1]
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			vecInd[i] = (individual.getObjective(i) - z_[i]) / (nz_[i] - z_[i]);

		scale = innerproduct(vecInd, lambda) / innerproduct(vecInd, lambda);
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			vecProj[i] = vecInd[i] - scale * lambda[i];

		distance = norm_vector(vecProj);

		return distance;
		
	}
	
	/**
	 * Calculate the perpendicular distance between the solution and reference line
	 * @param individual
	 * @param lambda
	 * @return
	 */
	public double calculateDistance2(Solution individual, double[] lambda) {
		
		double distance;
		double distanceSum = 0.0;
		
		double[] vecInd  	   = new double[problem_.getNumberOfObjectives()];
		double[] normalizedObj = new double[problem_.getNumberOfObjectives()];
		
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			distanceSum += individual.getObjective(i);
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			normalizedObj[i] = individual.getObjective(i) / distanceSum;
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			vecInd[i] = normalizedObj[i] - lambda[i];
		
		distance = norm_vector(vecInd);

		return distance;
	}
	
	public double calculateDistance3(Solution indiv, double[] lambda) {

		// normalize the weight vector (line segment)
		double nd = norm_vector(lambda);
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			lambda[i] = lambda[i] / nd;

		double[] realA = new double[problem_.getNumberOfObjectives()];
		double[] realB = new double[problem_.getNumberOfObjectives()];

		// difference between current point and reference point
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			realA[i] = (indiv.getObjective(i) - z_[i]);

		// distance along the line segment
		double d1 = Math.abs(innerproduct(realA, lambda));

		// distance to the line segment
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			realB[i] = (indiv.getObjective(i) - (z_[i] + d1 * lambda[i]));

		double distance = norm_vector(realB);

		return distance;
	}

	/**
	 * Initialize the weight vectors for subproblems (We only use the data that are already available)
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
	 * Compute the utility of subproblems
	 */
	public void comp_utility() {
		
		double f1, f2, uti, delta;
		
		for (int i = 0; i < populationSize_; i++) {
			f1 = fitnessFunction(population_.get(i), lambda_[i]);
			f2 = fitnessFunction(savedValues_[i], lambda_[i]);
			
			delta = f2 - f1;
			if (delta > 0.001)
				utility_[i] = 1.0;
			else {
				uti = (0.95 + (0.05 * delta / 0.001)) * utility_[i];
				utility_[i] = uti < 1.0 ? uti : 1.0;
			}
			savedValues_[i] = new Solution(population_.get(i));
		}
	}

	/**
	 * Initialize the neighborhood of subproblems
	 */
	public void initNeighborhood() {
		double[] x = new double[populationSize_];
		int[] idx = new int[populationSize_];

		for (int i = 0; i < populationSize_; i++) {
			// calculate the distances based on weight vectors
			for (int j = 0; j < populationSize_; j++) {
				x[j] = Utils.distVector(lambda_[i], lambda_[j]);
				idx[j] = j;
			} // for

			// find 'niche' nearest neighboring subproblems
			Utils.minFastSort(x, idx, populationSize_, T_);

			for (int k = 0; k < T_; k++) {
				neighborhood_[i][k] = idx[k];
			}
		} // for
	} // initNeighborhood

  /**
   *
   */
  public void initPopulation() throws JMException, ClassNotFoundException {
    for (int i = 0; i < populationSize_; i++) {
      Solution newSolution = new Solution(problem_);

      problem_.evaluate(newSolution);
      evaluations_++;
      population_.add(newSolution) ;
      savedValues_[i] = new Solution(newSolution);
    } // for
  } // initPopulation

  	/**
  	 * Initialize the ideal point
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
	 * Initialize the nadir point
	 * @throws JMException
	 * @throws ClassNotFoundException
	 */
	void initNadirPoint() throws JMException, ClassNotFoundException {
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			nz_[i] = -1.0e+30;

		for (int i = 0; i < populationSize_; i++)
			updateNadirPoint(population_.get(i));
	} // initNadirPoint

	/**
  	 * Mating selection is used to select the mating parents for offspring generation
  	 * @param list : the set of the indexes of selected mating parents
  	 * @param cid  : the id of current subproblem
  	 * @param size : the number of selected mating parents
  	 * @param type : 1 - neighborhood; otherwise - whole population
  	 */
	public Solution[] matingSelection(Vector<Integer> list, int cid, int size, int type) {
		
		int ss, r, p;
		
		Solution[] parents = new Solution[3];
		
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
		
		parents[0] = population_.get(list.get(0));
		parents[1] = population_.get(list.get(1));
		parents[2] = population_.get(cid);
				
		return parents;
	} // matingSelection

	/**
	 * Check the Pareto dominance relationship between two solutions
	 * @param a
	 * @param b
	 * @return
	 */
	public int Check_Dominance(Solution a, Solution b) {
		int[] flag1 = new int[1];
		int[] flag2 = new int[1];

		for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
			if (a.getObjective(i) < b.getObjective(i)) {
				flag1[0] = 1;
			} else {
				if (a.getObjective(i) > b.getObjective(i)) {
					flag2[0] = 1;
				}
			}
		}
		if (flag1[0] == 1 && flag2[0] == 0) {
			return 1;
		} else {
			if (flag1[0] == 0 && flag2[0] == 1) {
				return -1;
			} else {
				return 0;
			}
		}
	}

	public List<Integer> tour_selection(int depth) {

		// selection based on utility
		List<Integer> selected = new ArrayList<Integer>();
		List<Integer> candidate = new ArrayList<Integer>();

		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			selected.add(i); // WARNING! HERE YOU HAVE TO USE THE WEIGHT
								// PROVIDED BY QINGFU (NOT SORTED!!!!)

		for (int i = problem_.getNumberOfObjectives(); i < populationSize_; i++)
			candidate.add(i); // set of unselected weights

		while (selected.size() < (int) (populationSize_ / 5.0)) {
			// int best_idd = (int) (rnd_uni(&rnd_uni_init)*candidate.size()),
			// i2;
			int best_idd = (int) (PseudoRandom.randDouble() * candidate.size());
			// System.out.println(best_idd);
			int i2;
			int best_sub = candidate.get(best_idd);
			int s2;
			for (int i = 1; i < depth; i++) {
				i2 = (int) (PseudoRandom.randDouble() * candidate.size());
				s2 = candidate.get(i2);
				// System.out.println("Candidate: "+i2);
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
   	 * Update the ideal point, it is just an approximation with the best value for each objective
   	 * @param individual
   	 */
	void updateReference(Solution individual) {
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
			if (individual.getObjective(i) < z_[i])
				z_[i] = individual.getObjective(i);
		}
	} // updateReference
  
  	/**
  	 * Update the nadir point, it is just an approximation with worst value for each objective
  	 * @param individual
  	 */
	void updateNadirPoint(Solution individual) {
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
			if (individual.getObjective(i) > nz_[i])
				nz_[i] = individual.getObjective(i);
		}
	} // updateNadirPoint
	
	/**
	 * Calculate the dot product of two vectors
	 * @param vec1
	 * @param vec2
	 * @return
	 */
	public double innerproduct(double[] vec1, double[] vec2) {
		double sum = 0;
		
		for (int i = 0; i < vec1.length; i++)
			sum += vec1[i] * vec2[i];
		
		return sum;
	}

	/**
	 * Calculate the norm of the vector
	 * @param z
	 * @return
	 */
	public double norm_vector(double[] z) {
		double sum = 0;
		
		for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
			sum += z[i] * z[i];
		
		return Math.sqrt(sum);
	}

	/**
	 * Calculate the fitness value of a given individual, based on the specific scalarizing function
	 * @param individual
	 * @param lambda
	 * @return
	 */
	double fitnessFunction(Solution individual, double[] lambda) {
		double fitness;
		fitness = 0.0;

		if (functionType_.equals("_TCHE1")) {
			double maxFun = -1.0e+30;

			for (int i = 0; i < problem_.getNumberOfObjectives(); i++) {
				double diff = Math.abs(individual.getObjective(i) - z_[i]);

				double feval;
				if (lambda[i] == 0) {
					feval = 0.000001 * diff;
				} else {
					feval = diff * lambda[i];
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
		} else if (functionType_.equals("_PBI")) {
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
  
  
  /** @author Juanjo
   * This method selects N solutions from a set M, where N <= M
   * using the same method proposed by Qingfu Zhang, W. Liu, and Hui Li in
   * the paper describing MOEA/D-DRA (CEC 09 COMPTETITION)
   * An example is giving in that paper for two objectives. 
   * If N = 100, then the best solutions  attenting to the weights (0,1), 
   * (1/99,98/99), ...,(98/99,1/99), (1,0) are selected. 
   * 
   * Using this method result in 101 solutions instead of 100. We will just 
   * compute 100 even distributed weights and used them. The result is the same
   * 
   * In case of more than two objectives the procedure is:
   * 1- Select a solution at random
   * 2- Select the solution from the population which have maximum distance to
   * it (whithout considering the already included)
   * 
   * 
   * 
   * @param n: The number of solutions to return
   * @return A solution set containing those elements
   * 
   */
	SolutionSet finalSelection(int n) throws JMException {
		SolutionSet res = new SolutionSet(n);
		if (problem_.getNumberOfObjectives() == 2) { // subcase 1
			double[][] intern_lambda = new double[n][2];
			for (int i = 0; i < n; i++) {
				double a = 1.0 * i / (n - 1);
				intern_lambda[i][0] = a;
				intern_lambda[i][1] = 1 - a;
			} // for

			// we have now the weights, now select the best solution for each of
			// them
			for (int i = 0; i < n; i++) {
				Solution current_best = population_.get(0);
				int index = 0;
				double value = fitnessFunction(current_best, intern_lambda[i]);
				for (int j = 1; j < n; j++) {
					double aux = fitnessFunction(population_.get(j),
							intern_lambda[i]); // we are looking the best for
												// the weight i
					if (aux < value) { // solution in position j is better!
						value = aux;
						current_best = population_.get(j);
					}
				}
				res.add(new Solution(current_best));
			}

		} else { // general case (more than two objectives)

			Distance distance_utility = new Distance();
			int random_index = PseudoRandom.randInt(0, population_.size() - 1);

			// create a list containing all the solutions but the selected one
			// (only references to them)
			List<Solution> candidate = new LinkedList<Solution>();
			candidate.add(population_.get(random_index));

			for (int i = 0; i < population_.size(); i++) {
				if (i != random_index)
					candidate.add(population_.get(i));
			} // for

			while (res.size() < n) {
				int index = 0;
				Solution selected = candidate.get(0); // it should be a next! (n
														// <= population size!)
				double distance_value = distance_utility
						.distanceToSolutionSetInObjectiveSpace(selected, res);
				int i = 1;
				while (i < candidate.size()) {
					Solution next_candidate = candidate.get(i);
					double aux = distance_value = distance_utility
							.distanceToSolutionSetInObjectiveSpace(
									next_candidate, res);
					if (aux > distance_value) {
						distance_value = aux;
						index = i;
					}
					i++;
				}

				// add the selected to res and remove from candidate list
				res.add(new Solution(candidate.remove(index)));
			} //
		}
		return res;
	}
} // MOEA/D-IR
